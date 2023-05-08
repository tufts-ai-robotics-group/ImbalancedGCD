import argparse
import json
from pathlib import Path
import random

# from sklearn.metrics import roc_auc_score
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from gcd_data.get_datasets import get_class_splits, get_datasets, get_imbalanced_datasets

from imbalanced_gcd.model import DinoGCD
from imbalanced_gcd.augmentation import train_twofold_transform, DINOTestTrans
from imbalanced_gcd.test.eval import evaluate, cache_test_outputs, eval_from_cache
from imbalanced_gcd.loss import GCDLoss
from imbalanced_gcd.logger import AverageWriter


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_e", type=float, default=5e-5,
                        help="Learning rate for embedding v(x)")
    parser.add_argument("--lr_c", type=float, default=1e-2,
                        help="Learning rate for linear classifier {w_y, b_y}")
    parser.add_argument("--num_workers", type=int, default=4)
    # clustering hyperparameters
    parser.add_argument("--num_bootstrap", type=int, default=100,
                        help="Number of bootstrap rounds for clustering")
    parser.add_argument("--ss_method", type=str, default="KMeans",
                        choices=["KMeans", "GMM"],
                        help="Semi supervised clustering method. options: KMeans, GMM")
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
    # TODO: Mention how to get the splits
    if args.imbalance_method is None:
        train_dataset, valid_dataset, _ = get_datasets(args.dataset_name,
                                                       train_twofold_transform(
                                                           args.image_size),
                                                       DINOTestTrans(args.image_size),
                                                       args)[:3]
        test_dataset, _, _ = get_datasets(args.dataset_name,
                                          DINOTestTrans(args.image_size),
                                          None,
                                          args)[:3]
    else:
        train_dataset, valid_dataset, _ = get_imbalanced_datasets(args.dataset_name,
                                                                  train_twofold_transform(
                                                                      args.image_size),
                                                                  DINOTestTrans(args.image_size),
                                                                  args)[:3]
        test_dataset, _, _ = get_imbalanced_datasets(args.dataset_name,
                                                     DINOTestTrans(args.image_size),
                                                     None,
                                                     args)[:3]

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
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    valid_loader = DataLoader(valid_dataset, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **dataloader_kwargs)
    return train_loader, valid_loader, test_loader, args


def train_gcd(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    train_loader, valid_loader, test_loader, args = get_gcd_dataloaders(args)

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
    # init loss
    loss_func = GCDLoss(normal_classes, args.sup_weight)
    # init tensorboard, with random comment to stop overlapping runs
    av_writer = AverageWriter(args.label, comment=str(random.randint(0, 9999)))
    out_dir = Path(av_writer.writer.get_logdir())
    # metric dict for recording hparam metrics
    metric_dict = {}
    # save args
    keys = ['dataset_name', 'prop_train_label', 'imbalance_method', 'imbalance_ratio',
            'prop_minority_class', 'seed', 'label', 'num_epocs', 'batch_size', 'lr_e',
            'lr_c', 'sup_weight']
    args_dict = {k: v for k, v in vars(args).items() if k in keys}
    with open(Path(av_writer.writer.get_logdir()) / "args.json", "w") as f:
        json.dump(args_dict, f)
    # model training
    for epoch in range(args.num_epochs):
        # The validation and test dataloaders will be used only at the last epoch
        phase = 'Train'
        model.train()
        for (t_data, data), targets, uq_idx, label_mask in train_loader:
            data = data.to(device)
            t_data = t_data.to(device)
            targets = targets.long().to(device)
            label_mask = label_mask.to(device)
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                embeds = model(data)
                t_embeds = model(t_data)
                loss = loss_func(embeds, t_embeds, targets, label_mask)
                loss.backward()
                optim.step()
                scheduler.step()
                # only report train loss
                av_writer.update(f"{phase}/Average Loss", loss, torch.sum(label_mask))
        # # record end of training stats, grouped as Metrics in Tensorboard
        # # note non-numeric values (NaN, None, ect.) will cause entry
        # # to not be displayed in Tensorboard HPARAMS tab
        # # record metrics in last epoch
        # if epoch == args.num_epochs - 1:
        #     phases = ["Valid", "Test"]
        #     print('Evaluating on validation and test sets...')
        #     for phase in phases:
        #         if phase == "Valid":
        #             model.eval()
        #             dataloader = valid_loader
        #         else:
        #             model.eval()
        #             dataloader = test_loader
        #         epoch_embeds = torch.empty((0, model.out_dim)).to(device)
        #         epoch_targets = torch.empty((0,)).to(device)
        #         epoch_label_mask = torch.empty((0,), dtype=torch.bool).to(device)
        #         for batch in dataloader:
        #             if phase == "Valid":
        #                 data, targets, uq_idx = batch
        #                 label_mask = torch.isin(targets, normal_classes)
        #             else:  # Test phase
        #                 data, targets, uq_idx, label_mask = batch
        #             data = data.to(device)
        #             targets = targets.long().to(device)
        #             label_mask = label_mask.to(device)
        #             embeds = model(data)
        #             epoch_embeds = torch.vstack((epoch_embeds, embeds))
        #             epoch_targets = torch.hstack((epoch_targets, targets))
        #             epoch_label_mask = torch.hstack((epoch_label_mask, label_mask))
        #         # calculate accuracy
        #         overall, normal, novel, auroc = evaluate(args, out_dir, epoch_embeds,
        #                                                  epoch_targets, epoch_label_mask,
        #                                                  args.ss_method, args.num_bootstrap)
        #         tag = ['overall', 'normal', 'novel']
        #         acc_list = [overall, normal, novel]
        #         for i, t in enumerate(tag):
        #             av_writer.update(f"{phase}/Accuracy/{t}", acc_list[i][0])
        #             av_writer.update(f"{phase}/Accuracy/{t}_ci_low", acc_list[i][1])
        #             av_writer.update(f"{phase}/Accuracy/{t}_ci_high", acc_list[i][2])
        #             av_writer.update(f"{phase}/AUROC/{t}", auroc[i])
        # output statistics
        av_writer.write(epoch)
    # record hparams all at once and after all other writer calls
    # to avoid issues with Tensorboard changing output file
    av_writer.writer.add_hparams({
        "lr_e": args.lr_e,
        "lr_c": args.lr_c,
        "sup_weight": args.sup_weight,
    }, metric_dict)
    torch.save(model.state_dict(), out_dir / f"{args.num_epochs}.pt")
    cache_test_outputs(model, normal_classes, test_loader, out_dir)
    # record end of training stats, grouped as Metrics in Tensorboard
    # note non-numeric values (NaN, None, ect.) will cause entry
    # to not be displayed in Tensorboard HPARAMS tab
    # record metrics in last epoch
    print('Evaluating on the test set...')
    overall, normal, novel, auroc = eval_from_cache(args, out_dir)
    tag = ['overall', 'normal', 'novel']
    acc_list = [overall, normal, novel]
    for i, t in enumerate(tag):
        av_writer.update(f"{phase}/Accuracy/{t}", acc_list[i][0])
        av_writer.update(f"{phase}/Accuracy/{t}_ci_low", acc_list[i][1])
        av_writer.update(f"{phase}/Accuracy/{t}_ci_high", acc_list[i][2])
        av_writer.update(f"{phase}/AUROC/{t}", auroc[i])


if __name__ == "__main__":
    args = get_args()
    train_gcd(args)
