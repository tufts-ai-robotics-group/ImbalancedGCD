from imbalanced_gcd.ss_kmeans import SSKMeans
import imbalanced_gcd.eval.stats as stats

import torch
import numpy as np


@torch.no_grad()
def calc_accuracy(model, labeled_loader, unlabeled_loader, args, embedding_ind=None):
    model.eval()
    # get dataloader
    device = args.device
    # collect labeled embeddings and labels
    labeled_embeddings = np.empty((0, args.num_labeled_classes))
    labeled_y = np.empty((0,))
    for data, targets, uq_idxs in labeled_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        if embedding_ind is None:
            data_embeddings = outputs.detach().cpu().numpy()
        else:
            data_embeddings = outputs[embedding_ind].detach().cpu().numpy()
        labeled_embeddings = np.vstack((labeled_embeddings, data_embeddings))
        labeled_y = np.hstack((labeled_y, targets.cpu().numpy()))
    # collect unlabeled embeddings and labels
    embeddings = np.empty((0, args.num_labeled_classes))
    y_true = np.empty((0,))
    for data, targets, uq_idxs in unlabeled_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        if embedding_ind is None:
            data_embeddings = outputs.detach().cpu().numpy()
        else:
            data_embeddings = outputs[embedding_ind].detach().cpu().numpy()
        embeddings = np.vstack((embeddings, data_embeddings))
        y_true = np.hstack((y_true, targets.cpu().numpy()))
        # apply SS clustering and output results
        # SS KMeans
    ss_est = SSKMeans(labeled_embeddings, labeled_y,
                      (args.num_unlabeled_classes + args.num_labeled_classes)).fit(
        embeddings)
    y_pred = ss_est.predict(embeddings)
    # calculate accuracy
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)

    return acc
