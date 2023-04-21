import torch
import torch.nn as nn


class GCDLoss(nn.Module):
    def __init__(self, norm_targets, supervised_weight=.35):
        super().__init__()
        self.sup_weight = supervised_weight
        self.num_norm_targets = len(norm_targets)
        self.unsup_temp = .1
        self.sup_temp = .1

    def forward(self, embeds, t_embeds, targets):
        if targets.size() == torch.Size([0]):
            return torch.tensor(0., requires_grad=True).to(targets.device)
        else:
            # select labeled images according to the class reordering of base_dataset
            norm_mask = targets < self.num_norm_targets
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
