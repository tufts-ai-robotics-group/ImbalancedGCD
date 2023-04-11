import argparse
from pathlib import Path
import random

# from sklearn.metrics import roc_auc_score
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gcd_data.get_datasets import get_class_splits, get_datasets, get_imbalanced_datasets

from imbalanced_gcd.model import DinoGCD
from imbalanced_gcd.augmentation import gcd_twofold_transform
from imbalanced_gcd.eval.eval import calc_accuracy
from imbalanced_gcd.loss import GCDLoss
from imbalanced_gcd.ss_kmeans import SSKMeans


def get_args():
    parser = argparse.ArgumentParser()
    # dataset and split arguments
    parser.add_argument(
        "--dataset_name", type=str, default="cub",
        choices=["NovelCraft", "cifar10", "cifar100", "imagenet_100", "cub", "scars",
                 "fgvc_aricraft", "herbarium_19"],
        help="options: NovelCraft, cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, " +
             "herbarium_19")
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--imbalance_method', type=str, default=None, help='options:linear, step')
    parser.add_argument('--imbalance_ratio', type=float, default=2)
    parser.add_argument('--prop_minority_class', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)

    # model label for logging
    parser.add_argument("--label", type=str, default=None)
    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_e", type=float, default=5e-5,
                        help="Learning rate for embedding v(x)")
    parser.add_argument("--lr_c", type=float, default=1e-2,
                        help="Learning rate for linear classifier {w_y, b_y}")
    parser.add_argument("--num_workers", type=int, default=4)
    # loss hyperparameters
    parser.add_argument("--sup_weight", type=float, default=0.35,
                        help="Supervised loss weight")
    args = parser.parse_args()
    # prepend runs folder to label if given
    if args.label is not None:
        args.label = "runs/" + args.label
    # adjust learning rates based on batch size
    args.lr_e *= args.batch_size / 256
    args.lr_c *= args.batch_size / 256
    return args


def get_gcd_dataloaders(args):
    """Get generalized category discovery DataLoaders
    Args:
        args (Namespace): args containing dataset and prop_train_labels
        transforms (bool, optional): Whether to use train and test transforms. Defaults to False.
    Returns:
        tuple: train_loader: Normal training set
            valid_loader: Normal and novel validation set
            test_loader: Normal and novel test set
            args: args updated with num_labeled_classes and num_unlabeled_classes
    """
    args = get_class_splits(args)
    if args.imbalance_method is None:
        dataset_dict = get_datasets(args.dataset_name,
                                    gcd_twofold_transform(args.image_size),
                                    gcd_twofold_transform(args.image_size),
                                    args)[-1]
    else:
        dataset_dict = get_imbalanced_datasets(args.dataset_name,
                                               gcd_twofold_transform(args.image_size),
                                               gcd_twofold_transform(args.image_size),
                                               args)[-1]
    # add number of labeled and unlabeled classes to args
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    # construct DataLoaders
    # need to set num_workers=0 in Windows due to torch.multiprocessing pickling limitation
    generator = torch.Generator()
    generator.manual_seed(0)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
        "generator": generator
    }
    train_loader = DataLoader(dataset_dict["train_labeled"], **dataloader_kwargs)
    valid_loader = DataLoader(dataset_dict["test"], **dataloader_kwargs)
    test_loader = DataLoader(dataset_dict["train_unlabeled"], **dataloader_kwargs)
    return train_loader, valid_loader, test_loader, args


def train_gcd(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    train_loader, valid_loader, test_loader, _ = get_gcd_dataloaders(args)

    # normal classes after target transform according to GCDdatasets API
    normal_classes = torch.arange(args.num_labeled_classes).to(device)
    args.normal_classes = normal_classes
    # init model
    model = DinoGCD().to(device)
    # init optimizer
    optim = torch.optim.AdamW([
        {
            "params": model.dino.parameters(),
            "lr": args.lr_e,
            "weignt_decay": 5e-4,
        },
        {
            "params": model.mlp.parameters(),
            "lr": args.lr_c,
        },
    ])
    # set learning rate warmup to take 1/4 of training time
    warmup_epochs = max(args.num_epochs // 4, 1)
    # init learning rate scheduler
    warmup_iters = warmup_epochs * len(train_loader)
    total_iters = args.num_epochs * len(train_loader)
    scheduler = lr_scheduler.SequentialLR(
        optim,
        [
            lr_scheduler.LinearLR(optim, start_factor=1 / warmup_iters, total_iters=warmup_iters),
            lr_scheduler.CosineAnnealingLR(optim, total_iters - warmup_iters)
        ],
        [warmup_iters])
    phases = ["Train", "Valid", "Test"]
    # init loss
    loss_func = GCDLoss(normal_classes, args.sup_weight)
    # init tensorboard, with random comment to stop overlapping runs
    writer = SummaryWriter(args.label, comment=str(random.randint(0, 9999)))
    # cache labeled training data for SSKM
    train_labeled_data = torch.empty(0, 3, args.image_size, args.image_size).to(device)
    train_labeled_targets = torch.empty(0, dtype=torch.long).to(device)
    for (t_data, data), targets, uq_idx in train_loader:
        train_labeled_data = torch.vstack((train_labeled_data, data.to(device)))
        train_labeled_targets = torch.hstack((train_labeled_targets, targets.to(device)))
    # metric dict for recording hparam metrics
    metric_dict = {}
    # model training
    from tqdm import tqdm
    for epoch in tqdm(range(args.num_epochs)):
        # Each epoch has a training, validation, and test phase
        for phase in phases:
            if phase == "Train":
                model.train()
                dataloader = train_loader
            elif phase == "Valid":
                model.eval()
                dataloader = valid_loader
            else:
                model.eval()
                dataloader = test_loader
            # vars for tensorboard stats
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            # tensors for caching embeddings and targets
            for (t_data, data), targets, uq_idx in dataloader:
                # forward and loss
                data = data.to(device)
                targets = targets.long().to(device)
                t_data = t_data.to(device)
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "Train"):
                    embeds = model(data)
                    t_embeds = model(t_data)
                    if phase == "Train":
                        # all true mask if training
                        norm_mask = torch.ones((data.size(0),), dtype=torch.bool).to(device)
                    else:
                        # filter out novel examples from loss in non-training phases
                        norm_mask = torch.isin(targets, normal_classes).to(device)
                    # make sure the mask has at least one True
                    if torch.any(norm_mask):
                        loss = loss_func(embeds[norm_mask], t_embeds[norm_mask], targets[norm_mask])
                    else:
                        loss = torch.tensor(0.).to(device)
                # backward and optimize only if in training phase
                if phase == "Train":
                    loss.backward()
                    optim.step()
                    scheduler.step()
                # calculate statistics
                epoch_loss = (loss.item() * data.size(0) +
                              cnt * epoch_loss) / (cnt + data.size(0))
                cnt += data.size(0)
            # output statistics
            writer.add_scalar(f"{phase}/Average Loss", epoch_loss, epoch)
            epoch_embeds = model(train_labeled_data)
            epoch_acc = calc_accuracy(model, args, epoch_embeds, train_labeled_targets, test_loader)
            writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)
            if phase != "Train":
                # record end of training stats, grouped as Metrics in Tensorboard
                if epoch == args.num_epochs - 1:
                    # note non-numeric values (NaN, None, ect.) will cause entry
                    # to not be displayed in Tensorboard HPARAMS tab
                    metric_dict.update({
                        f"Metrics/{phase}_loss": epoch_loss,
                        f"Metrics/{phase}_acc": epoch_acc,
                    })
    # record hparams all at once and after all other writer calls
    # to avoid issues with Tensorboard changing output file
    writer.add_hparams({
        "lr_e": args.lr_e,
        "lr_c": args.lr_c,
        "sup_weight": args.sup_weight,
    }, metric_dict)
    torch.save(model.state_dict(), Path(writer.get_logdir()) / f"{args.num_epochs}.pt")


if __name__ == "__main__":
    args = get_args()
    train_gcd(args)
