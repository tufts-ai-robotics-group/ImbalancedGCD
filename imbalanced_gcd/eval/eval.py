from imbalanced_gcd.ss_kmeans import SSKMeans
import imbalanced_gcd.eval.stats as stats

import torch
import numpy as np


@torch.no_grad()
def calc_accuracy(model, args, epoch_embeds, epoch_targets, dataloader, phase):
    model.eval()
    # get dataloader
    device = args.device
    # collect unlabeled embeddings and labels
    embeddings = np.empty((0, model.out_dim))
    y_true = np.empty((0,))
    # for (t_data, data), targets, uq_idxs in dataloader:
    for batch in dataloader:
        if phase == 'train':
            (t_data, data), targets, uq_idxs, label_mask = batch
        else:
            (t_data, data), targets, uq_idxs = batch
            label_mask = torch.zeros_like(targets, dtype=torch.bool)
        data, targets = data.to(device), targets.to(device)
        outputs = model(data[~label_mask])
        data_embeddings = outputs.detach().cpu().numpy()
        embeddings = np.vstack((embeddings, data_embeddings))
        y_true = np.hstack((y_true, targets[~label_mask].cpu().numpy()))
    # apply SS clustering and output results
    # SS KMeans
    epoch_embeds = epoch_embeds.detach().cpu().numpy()
    epoch_targets = epoch_targets.detach().cpu().numpy()
    embeddings = embeddings.astype(epoch_embeds.dtype)
    ss_est = SSKMeans(epoch_embeds, epoch_targets,
                      (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
        embeddings)
    y_pred = ss_est.predict(embeddings)
    # calculate accuracy
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)

    return acc
