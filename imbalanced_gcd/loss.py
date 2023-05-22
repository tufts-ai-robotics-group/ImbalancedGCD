import torch
import torch.nn as nn


class GCDLoss(nn.Module):
    def __init__(self, norm_targets, supervised_weight=.35):
        super().__init__()
        self.sup_weight = supervised_weight
        self.num_norm_targets = len(norm_targets)
        self.unsup_temp = .1
        self.sup_temp = .1

    def forward(self, embeds, t_embeds, targets, norm_mask):
        embeds = embeds.reshape(-1, embeds.shape[1])
        t_embeds = t_embeds.reshape(-1, t_embeds.shape[1])
        if targets.size() == torch.Size([0]):
            return torch.tensor(0., requires_grad=True).to(targets.device)
        else:
            unsup_loss = self.unsup_contrast_loss(embeds, t_embeds)
            sup_loss = self.sup_contrast_loss(embeds[norm_mask], targets[norm_mask])
            return (1 - self.sup_weight) * unsup_loss + self.sup_weight * sup_loss

    def dot_others(self, embeds):
        # return dot product with each other embedding, excluding self * self
        # output is N x N - 1 due to exclusion
        n = embeds.shape[0]
        return (embeds @ embeds.T)[~torch.eye(n, dtype=bool)].reshape((n, n - 1))

    def unsup_contrast_loss(self, embeds, t_embeds):
        if embeds.shape[0] < 2:
            return torch.tensor(0., requires_grad=True).to(embeds.device)
        # dot product of t ransformed image over dot product over other images
        nums = torch.sum(embeds * t_embeds, dim=1) / self.unsup_temp
        denom_prods = self.dot_others(embeds) / self.unsup_temp
        denoms = torch.logsumexp(denom_prods, dim=1)
        losses = denoms - nums
        return torch.mean(losses)

    def sup_contrast_loss(self, embeds, targets):
        if embeds.shape[0] < 2:
            return torch.tensor(0., requires_grad=True).to(embeds.device)
        unique_targets = torch.unique(targets)
        losses = torch.Tensor([]).to(embeds.device)
        # denominator calculation is the same regardless of class
        denom_prods = self.dot_others(embeds) / self.sup_temp
        denoms = torch.logsumexp(denom_prods, dim=1)
        for target in unique_targets:
            # dot product of same class over dot product over all classes
            target_mask = targets == target
            if torch.sum(target_mask) == 1:
                continue
            target_embeds = embeds[target_mask]
            target_denoms = denoms[target_mask]
            nums = torch.sum(self.dot_others(target_embeds) / self.sup_temp, dim=1)
            # only dividing nums since otherwise target_denoms needs to be repeated in above sum
            losses = torch.cat((losses, target_denoms - (nums / len(nums))))
        if losses.size() == torch.Size([0]):
            return torch.tensor(0., requires_grad=True).to(embeds.device)
        else:
            return torch.mean(losses)


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def info_nce_logits(features, args):
    from torch.nn import functional as F
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = features.reshape(-1, features.shape[1])

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.unsup_temp
    return logits, labels


def get_args():
    import argparse
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


if __name__ == "__main__":
    # Test if GCDLoss and the combination of SupConLoss and info_nce_logits are equivalent
    # Make dummy data of features and targets
    import time
    from gcd_data.get_datasets import get_class_splits
    args = get_args()
    args = get_class_splits(args)
    args.unsup_temp = 0.1
    args.sup_temp = 0.1
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    args.n_views = 2
    features = torch.randn((100, 128, 1)).to(device)
    targets = torch.randint(0, 10, (100,)).to(device)
    norm_mask = torch.randint(0, 2, (100,)).to(device)
    norm_mask = norm_mask.bool()
    norm_targets = torch.unique(targets[norm_mask])
    # Test GCDLoss
    gcd_loss = GCDLoss(norm_targets)
    gcdloss = gcd_loss(features, features, targets, norm_mask)

    supcon_loss = SupConLoss()
    sup_loss = supcon_loss(features, targets)
    contrastive_logits, contrastive_labels = info_nce_logits(features, args)
    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

    original_loss = args.sup_weight * sup_loss + (1 - args.sup_weight) * contrastive_loss

    print(f"GCDLoss: {gcdloss}")
    print(f"Original loss: {original_loss}")
