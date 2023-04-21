from imbalanced_gcd.ss_kmeans import SSKMeans
from imbalanced_gcd.ss_gmm import SSGMM
import imbalanced_gcd.eval.stats as stats

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
    for (t_data, data), targets, uq_idx, label_mask in train_loader:
        if torch.any(label_mask):
            embeds = model(data[label_mask].to(device))
            train_labeled_embed = torch.vstack((train_labeled_embed, embeds))
            train_labeled_targets = torch.hstack((train_labeled_targets,
                                                  targets[label_mask].to(device)))
    train_labeled_embed = train_labeled_embed.detach().cpu().numpy()
    train_labeled_targets = train_labeled_targets.detach().cpu().numpy()
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
                epoch_embeds)
        elif ss_method == 'GMM':
            # SS GMM
            ss_est = SSGMM(train_labeled_embed, train_labeled_targets, epoch_embeds,
                           (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
                epoch_embeds)
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
