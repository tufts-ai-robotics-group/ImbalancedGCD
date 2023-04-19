import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_


class DinoGCD(nn.Module):
    def __init__(self, out_dim=128, use_bn=True,
                 norm_last_layer=True, nlayers=3, hidden_dim=512, bottleneck_dim=256) -> None:
        super().__init__()
        self.out_dim = out_dim
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.embed_len = self.dino.norm.normalized_shape[0]
        # linear classification head
        if nlayers == 1:
            self.mlp = nn.Linear(self.embed_len, bottleneck_dim)
        else:
            layers = [nn.Linear(self.embed_len, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # add one more linear layer to match the dimension of the output
        last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            last_layer.weight_g.requires_grad = False
        self.mlp.add_module('last_layer', last_layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # DINO embeddings
        raw_embeds = self.dino(x)
        embeds = raw_embeds.view(raw_embeds.shape[0], -1)
        # Embeddings after MLP that is used for classification
        embeds = self.mlp(embeds)
        return embeds
