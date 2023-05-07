from imbalanced_gcd.ss_kmeans import SSKMeans
from imbalanced_gcd.ss_gmm import SSGMM
import imbalanced_gcd.eval.stats as stats

from pathlib import Path
from sklearn.metrics import roc_auc_score, pairwise_distances_argmin_min
import torch
import numpy as np
import time


def evaluate(args, epoch_embeds, epoch_targets, label_mask,
             ss_method='KMeans', num_bootstrap=1):
    epoch_embeds = epoch_embeds.detach().cpu().numpy()
    epoch_targets = epoch_targets.detach().cpu().numpy()
    test_labeled_embed = epoch_embeds[label_mask]
    test_labeled_targets = epoch_targets[label_mask]
    test_unlabeled_embed = epoch_embeds[~label_mask]
    norm_embeds = torch.isin(epoch_targets, args.normal_classes).detach().cpu().numpy()
    # apply SS clustering and output results
    # initialize accuracy array
    overall_acc = np.zeros(num_bootstrap)
    normal_acc = np.zeros(num_bootstrap)
    novel_acc = np.zeros(num_bootstrap)
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
    y_pred = ss_est.predict(test_unlabeled_embed)
    # calculate overall accuracy
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, epoch_targets)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    overall_acc[i] = acc
    # calculate normal accuracy
    row_ind, col_ind, weight = stats.assign_clusters(y_pred[norm_embeds],
                                                     epoch_targets[norm_embeds])
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    normal_acc[i] = acc
    # calculate novel accuracy
    row_ind, col_ind, weight = stats.assign_clusters(y_pred[~norm_embeds],
                                                     epoch_targets[~norm_embeds])
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    novel_acc[i] = acc
    end = time.time()
    print(f'Average clustering time: {(end - start)/num_bootstrap:.2f} seconds')

    # compute AUROC
    auroc_list = calc_multiclass_auroc(ss_est, epoch_embeds, epoch_targets)

    # compute mean and confidence interval for overall, normal, and novel accuracy
    acc_list = np.array([overall_acc, normal_acc, novel_acc])
    acc_mean = np.mean(acc_list, axis=1)
    ci_low = np.percentile(acc_list, 2.5, axis=1)
    ci_high = np.percentile(acc_list, 97.5, axis=1)

    return (acc_mean, ci_low, ci_high), auroc_list


def cache_test_outputs(model, normal_classes, test_loader, out_dir):
    out_dir = Path(out_dir)
    model.eval()
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # initialize caches
    out_embeds = torch.empty((0, model.out_dim)).to(device)
    out_targets = torch.tensor([], dtype=int).to(device)
    out_norm_mask = torch.tensor([], dtype=bool).to(device)
    for (t_data, data), targets, uq_idx, norm_mask in test_loader:
        # move data to device
        data = data.to(device)
        targets = targets.long().to(device)
        norm_mask = norm_mask.to(device)
        # forward pass
        with torch.set_grad_enabled(False):
            embeds = model(data)
        # cache data
        out_embeds = torch.vstack((out_embeds, embeds))
        out_targets = torch.hstack((out_targets, targets))
        out_norm_mask = torch.hstack((out_norm_mask, norm_mask))
    # write caches
    torch.save(out_embeds.cpu(), out_dir / "embeds.pt")
    torch.save(out_targets.cpu(), out_dir / "targets.pt")
    torch.save(out_norm_mask.cpu(), out_dir / "norm_mask.pt")


def calc_multiclass_auroc(ss_est, embeds, targets, norm_mask):
    """
    Multi-class AUROC calculation based on a KMeans estimator.
    The class probabilities are calculated as the negative distance to the cluster center.

    :param ss_est: SSKMeans/SSGMM estimator that has been fit to the training set
    :param embeds (np.array): embeddings of the test set
    :param targets (np.array): targets of the test set
    :param norm_mask (np.array): mask of normal classes in the test set
    """

    # calculate class probabilities
    class_dist = ss_est.transform(embeds)
    # closer class is more likely. use softmax to convert to probabilities
    class_prob = torch.softmax(torch.tensor(-class_dist), dim=1)
    # get the class label for each centroid
    centroids, _ = pairwise_distances_argmin_min(ss_est.cluster_centers_, embeds)
    centroids_targets = targets[centroids]
    # check that each centroid is assigned to a unique class
    assert len(np.unique(centroids_targets)) == len(centroids_targets)
    # calculate AUROC for overall, normal, and novel classes
    # normalize the probabilities to 1 for normal and novel classes
    overall = roc_auc_score(targets, class_prob, multi_class='ovo', labels=centroids_targets)
    normal = roc_auc_score(targets[norm_mask],
                           class_prob[norm_mask] / class_prob[norm_mask].sum(axis=1),
                           multi_class='ovo',
                           labels=centroids_targets[norm_mask])
    novel = roc_auc_score(targets[~norm_mask],
                          class_prob[~norm_mask] / class_prob[~norm_mask].sum(axis=1),
                          multi_class='ovo',
                          labels=centroids_targets[~norm_mask])
    return overall, normal, novel
