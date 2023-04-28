from imbalanced_gcd.ss_kmeans import SSKMeans
from imbalanced_gcd.ss_gmm import SSGMM
import imbalanced_gcd.eval.stats as stats

from pathlib import Path
import torch
import numpy as np
import time


@torch.no_grad()
def calc_accuracy(model, args, train_loader, epoch_embeds, epoch_targets,
                  ss_method='KMeans', num_bootstrap=1):
    model.eval()
    device = args.device
    # collect labeled embeddings and labels in train_loader for SS clustering
    train_labeled_embed = torch.empty(0, model.out_dim).to(device)
    train_labeled_targets = torch.empty(0, dtype=torch.long).to(device)
    train_unlabeled_embed = torch.empty(0, model.out_dim).to(device)
    for (t_data, data), targets, uq_idx, label_mask in train_loader:
        if torch.any(label_mask):
            labeled_embeds = model(data[label_mask].to(device))
            unlabeled_embeds = model(data[~label_mask].to(device))
            train_labeled_embed = torch.vstack((train_labeled_embed, labeled_embeds))
            train_labeled_targets = torch.hstack((train_labeled_targets,
                                                  targets[label_mask].to(device)))
            train_unlabeled_embed = torch.vstack((train_unlabeled_embed, unlabeled_embeds))
    train_labeled_embed = train_labeled_embed.detach().cpu().numpy()
    train_labeled_targets = train_labeled_targets.detach().cpu().numpy()
    train_unlabeled_embed = train_unlabeled_embed.detach().cpu().numpy()
    epoch_embeds = epoch_embeds.detach().cpu().numpy()
    epoch_targets = epoch_targets.detach().cpu().numpy()
    # apply SS clustering and output results
    # initialize accuracy array
    acc_list = np.zeros(num_bootstrap)
    # record clustering time
    start = time.time()
    for i in range(num_bootstrap):
        if ss_method == 'KMeans':
            # SS KMeans
            ss_est = SSKMeans(train_labeled_embed, train_labeled_targets,
                              (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
                train_unlabeled_embed)
        elif ss_method == 'GMM':
            # SS GMM
            ss_est = SSGMM(train_labeled_embed, train_labeled_targets, train_unlabeled_embed,
                           (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
                train_unlabeled_embed)
        y_pred = ss_est.predict(epoch_embeds)
        # calculate accuracy
        row_ind, col_ind, weight = stats.assign_clusters(y_pred, epoch_targets)
        acc = stats.cluster_acc(row_ind, col_ind, weight)
        acc_list[i] = acc
    end = time.time()
    print(f'Average clustering time: {(end - start)/num_bootstrap:.2f} seconds')

    # compute mean and confidence interval
    acc_mean = np.mean(acc_list)
    ci_low = np.percentile(acc_list, 2.5)
    ci_high = np.percentile(acc_list, 97.5)

    return (acc_mean, ci_low, ci_high)


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
