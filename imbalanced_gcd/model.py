import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_


class DinoGCD(nn.Module):
    def __init__(self, out_dim=65536) -> None:
        super().__init__()
        self.out_dim = out_dim
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.embed_len = self.dino.norm.normalized_shape[0]
        self.mlp = DinoHead(in_dim=self.embed_len, out_dim=self.out_dim)

    def forward(self, x):
        # DINO embeddings
        raw_embeds = self.dino(x)
        embeds = raw_embeds.view(raw_embeds.shape[0], -1)
        # Embeddings after MLP that is used for classification
        embeds = self.mlp(embeds)
        return embeds


class DinoHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        # alternate linear layers and GELU activations, with no activation on last layer
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # layer with unit vector weights, should output unit vectors when given normalized inputs
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # freeze vector magnitude to 1
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # normalize vector so final outputs are unit vectors
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
