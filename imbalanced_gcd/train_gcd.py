import argparse
from pathlib import Path
import random

# from sklearn.metrics import roc_auc_score
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gcd_data.get_datasets import get_class_splits, get_datasets

from model import DinoGCD
from augmentation import sim_gcd_train, sim_gcd_test
from loss import GCDLoss


def get_args():
    parser = argparse.ArgumentParser()
    # dataset and split arguments
    parser.add_argument(
        "--dataset_name", type=str, default="cifar10",
        choices=["NovelCraft", "cifar10", "cifar100", "imagenet_100", "cub", "scars",
                 "fgvc_aricraft", "herbarium_19"],
        help="options: NovelCraft, cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, " +
             "herbarium_19")
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    # model label for logging
    parser.add_argument("--label", type=str, default=None)
    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr_e", type=float, default=1e-5,
                        help="Learning rate for embedding v(x)")
    parser.add_argument("--lr_c", type=float, default=1e-3,
                        help="Learning rate for linear classifier {w_y, b_y}")
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


def get_nd_dataloaders(args):
    """Get novelty detection DataLoaders
    Args:
        args (Namespace): args containing dataset and prop_train_labels
    Returns:
        tuple: train_loader: Normal training set
            valid_loader: Normal and novel validation set
            test_loader: Normal and novel test set
            args: args updated with num_labeled_classes and num_unlabeled_classes
    """
    # train_trans = sim_gcd_train()
    test_trans = sim_gcd_test()
    args = get_class_splits(args)
    dataset_dict = get_datasets(args.dataset_name, T.ToTensor(), test_trans, args)[-1]
    # add number of labeled and unlabeled classes to args
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    # construct DataLoaders
    # need to set num_workers=0 in Windows due to torch.multiprocessing pickling limitation
    dataloader_kwargs = {
        "batch_size": 1,
        "num_workers": 0,
        "shuffle": True,
    }
    train_loader = DataLoader(dataset_dict["train_labeled"], **dataloader_kwargs)
    valid_loader = DataLoader(dataset_dict["test"], **dataloader_kwargs)
    test_loader = DataLoader(dataset_dict["train_unlabeled"], **dataloader_kwargs)
    return train_loader, valid_loader, test_loader, args


def train_gcd(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # init dataloaders
    train_loader, valid_loader, test_loader, args = get_nd_dataloaders(args)
    # normal classes after target transform according to GCDdatasets API
    normal_classes = torch.arange(args.num_labeled_classes).to(device)
    # init model
    model = DinoGCD(args.num_labeled_classes).to(device)
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
    phases = ["train", "valid", "test"]
    # init loss
    loss_func = GCDLoss(normal_classes, args.sup_weight)
    # init tensorboard, with random comment to stop overlapping runs
    writer = SummaryWriter(args.label, comment=str(random.randint(0, 9999)))
    # metric dict for recording hparam metrics
    metric_dict = {}
    # set up train transform
    transforms = sim_gcd_train()
    # model training
    for epoch in range(args.num_epochs):
        # Each epoch has a training, validation, and test phase
        for phase in phases:
            if phase == "train":
                model.train()
                dataloader = train_loader
            elif phase == "valid":
                model.eval()
                dataloader = valid_loader
            else:
                model.eval()
                dataloader = test_loader
            # vars for tensorboard stats
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            for data, targets, uq_idxs in (dataloader):
                # forward and loss
                data = data.to(device)
                targets = targets.long().to(device)
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    embeds = model(data)
                    if phase == "train":
                        # all true mask if training
                        norm_mask = torch.ones((data.size(0),), dtype=torch.bool).to(device)
                    else:
                        # filter out novel examples from loss in non-training phases
                        norm_mask = torch.isin(targets, normal_classes).to(device)
                    # transform each data in the batch
                    transformed_data = []
                    for i in range(data.size(0)):
                        transformed_data.append(transforms((data[i])))
                        t_data = torch.stack(transformed_data).to(device)
                    t_embeds = model(t_data)
                    loss = loss_func(embeds[norm_mask], t_embeds[norm_mask], targets[norm_mask])
                # backward and optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optim.step()
                    scheduler.step()
                # calculate statistics
                _, preds = torch.max(embeds[norm_mask], 1)
                epoch_loss = (loss.item() * data.size(0) +
                              cnt * epoch_loss) / (cnt + data.size(0))
                if len(preds) > 0:
                    epoch_acc = (torch.sum(preds == targets[norm_mask].data) +
                                 epoch_acc * cnt).double() / (cnt + len(preds))
                cnt += data.size(0)
            # get phase label
            if phase == "train":
                phase_label = "Train"
            elif phase == "valid":
                phase_label = "Valid"
            else:
                phase_label = "Test"
            # output statistics
            writer.add_scalar(f"{phase_label}/Average Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase_label}/Average Accuracy", epoch_acc, epoch)
            if phase != "train":
                # record end of training stats, grouped as Metrics in Tensorboard
                if epoch == args.num_epochs - 1:
                    # note non-numeric values (NaN, None, ect.) will cause entry
                    # to not be displayed in Tensorboard HPARAMS tab
                    metric_dict.update({
                        f"Metrics/{phase_label}_loss": epoch_loss,
                        f"Metrics/{phase_label}_accuracy": epoch_acc,
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
