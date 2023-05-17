from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import time

from imbalanced_gcd.test.bootstrap import bootstrap_metric
from imbalanced_gcd.test.plot import plot_con_matrix, plot_gcd_ci
from imbalanced_gcd.test.stats import cluster_acc, cluster_confusion
from imbalanced_gcd.ss_kmeans import SSKMeans
from imbalanced_gcd.ss_gmm import SSGMM


def cache_test_outputs(model, normal_classes, test_loader, out_dir):
    out_dir = Path(out_dir)
    model.eval()
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # initialize caches
    out_embeds = torch.empty((0, model.out_dim)).to(device)
    out_targets = torch.tensor([], dtype=int).to(device)
    out_norm_mask = torch.tensor([], dtype=bool).to(device)
    out_label_mask = torch.tensor([], dtype=bool).to(device)
    for data, targets, uq_idxs, label_mask in test_loader:
        # move to device
        data = data.to(device)
        targets = targets.long().to(device)
        label_mask = label_mask.to(device)
        # create normal mask
        norm_mask = torch.isin(targets, normal_classes).to(device)
        # forward pass
        with torch.set_grad_enabled(False):
            embeds = model(data)
        # cache data
        out_embeds = torch.vstack((out_embeds, embeds))
        out_targets = torch.hstack((out_targets, targets))
        out_norm_mask = torch.hstack((out_norm_mask, norm_mask))
        out_label_mask = torch.hstack((out_label_mask, label_mask))
    # write caches
    torch.save(out_embeds.cpu(), out_dir / "embeds.pt")
    torch.save(out_targets.cpu(), out_dir / "targets.pt")
    torch.save(out_norm_mask.cpu(), out_dir / "norm_mask.pt")
    torch.save(out_label_mask.cpu(), out_dir / "label_mask.pt")


def evaluate(args, out_dir, epoch_embeds, epoch_targets, label_mask,
             ss_method='KMeans', num_bootstrap=1):
    epoch_embeds = epoch_embeds.detach().cpu().numpy()
    epoch_targets = epoch_targets.detach().cpu().numpy()
    test_labeled_embed = epoch_embeds[label_mask]
    test_unlabeled_embed = epoch_embeds[~label_mask]
    test_labeled_targets = epoch_targets[label_mask]
    test_unlabeled_targets = epoch_targets[~label_mask]
    # normal class mask for unlabeled test set
    norm_mask = torch.isin(test_unlabeled_targets, args.normal_classes).detach().cpu().numpy()
    # apply SS clustering and output results
    # record clustering time
    start = time.time()
    if ss_method == 'KMeans':
        # SS KMeans
        ss_est = SSKMeans(test_labeled_embed, test_labeled_targets,
                          (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
            test_unlabeled_embed)
    elif ss_method == 'GMM':
        # SS GMM
        ss_est = SSGMM(test_labeled_embed, test_labeled_targets, test_unlabeled_embed,
                       (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
            test_unlabeled_embed)
    end = time.time()
    print(f'Average clustering time: {(end - start)/num_bootstrap:.2f} seconds')

    y_pred = ss_est.predict(test_unlabeled_embed)
    y_true = test_unlabeled_targets
    # get accuracies
    overall = bootstrap_metric(y_pred, y_true, cluster_acc, n_bootstraps=num_bootstrap)
    normal = bootstrap_metric(y_pred[norm_mask], y_true[norm_mask], cluster_acc, num_bootstrap)
    novel = bootstrap_metric(y_pred[~norm_mask], y_true[~norm_mask], cluster_acc, num_bootstrap)
    plot_gcd_ci(overall, novel, normal).savefig(out_dir / "acc_ci.png")
    plot_con_matrix(cluster_confusion(y_pred, y_true)).savefig(out_dir / "conf_mat.png")

    # compute AUROC
    auroc_list = calc_multiclass_auroc(ss_est, epoch_embeds, epoch_targets, args)

    return (overall, normal, novel), auroc_list


def calc_multiclass_auroc(ss_est, embeds, targets, args):
    """
    Multi-class AUROC calculation based on a KMeans estimator.
    The class probabilities are calculated as the negative distance to the cluster center.

    :param ss_est: SSKMeans/SSGMM estimator that has been fit to the training set
    :param embeds (np.array): embeddings of the test set
    :param targets (np.array): targets of the test set
    :param args (argparse.Namespace): contains normal_classes that is used to create the normal mask
    """

    # calculate class probabilities
    class_dist = ss_est.transform(embeds)
    # closer class is more likely. use softmax to convert to probabilities
    class_prob = torch.softmax(torch.tensor(-class_dist), dim=1)
    # calculate AUROC for overall, normal, and novel classes
    # normalize the probabilities to 1 for normal and novel classes
    auroc_lists = roc_auc_score(targets, class_prob, multi_class='ovr',
                                average=None)
    overall = np.mean(auroc_lists)
    # apply mask to the vertical axis
    normal = np.mean(auroc_lists[:args.num_labeled_classes])
    novel = np.mean(auroc_lists[args.num_labeled_classes:])
    return [overall, normal, novel]


def eval_from_cache(args, out_dir):
    out_dir = Path(out_dir)
    # load caches
    embeds = torch.load(out_dir / "embeds.pt").numpy()
    targets = torch.load(out_dir / "targets.pt").numpy()
    norm_mask = torch.load(out_dir / "norm_mask.pt").numpy()
    label_mask = torch.load(out_dir / "label_mask.pt").numpy()
    # get unlabeled normal mask
    unlabel_norm_mask = np.logical_and(~label_mask, norm_mask)
    unlabel_novel_mask = np.logical_and(~label_mask, ~norm_mask)
    # apply SS clustering and output results
    # record clustering time
    start = time.time()
    if args.ss_method == 'KMeans':
        # SS KMeans
        ss_est = SSKMeans(embeds[label_mask], targets[label_mask],
                          (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
            embeds[~label_mask])
    elif args.ss_method == 'GMM':
        # SS GMM
        ss_est = SSGMM(embeds[label_mask], targets[label_mask], embeds[~label_mask],
                       (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
            embeds[~label_mask])
    end = time.time()
    print(f'Average clustering time: {(end - start):.2f} seconds')
    y_pred = ss_est.predict(embeds)
    y_true = targets
    # get accuracies
    overall = bootstrap_metric(y_pred[~label_mask], y_true[~label_mask],
                               cluster_acc, n_bootstraps=args.num_bootstrap)
    normal = bootstrap_metric(y_pred[unlabel_norm_mask], y_true[unlabel_norm_mask],
                              cluster_acc, args.num_bootstrap)
    novel = bootstrap_metric(y_pred[unlabel_novel_mask], y_true[unlabel_novel_mask],
                             cluster_acc, args.num_bootstrap)
    plot_gcd_ci(overall, novel, normal).savefig(out_dir / "acc_ci.png")
    plot_con_matrix(cluster_confusion(y_pred, y_true)).savefig(out_dir / "conf_mat.png")

    # compute AUROC
    auroc_list = calc_multiclass_auroc(ss_est, embeds[~label_mask],
                                       targets[~label_mask], args)

    return overall, normal, novel, auroc_list
