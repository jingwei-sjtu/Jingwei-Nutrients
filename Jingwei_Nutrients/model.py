import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FeatureTransformer(nn.Module):
    def __init__(self, embed_dim=64, output_dim=64, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transmodel = TransformerEncoder(layer, num_layers=num_layers)
        self.out_embed = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.transmodel(x)
        x = x.mean(dim=1)
        return self.out_embed(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.predictor = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.predictor(x)


class FusionModel(nn.Module):
    def __init__(
        self,
        num_features,
        embed_dim=64,
        embed_dim_foundation=64,
        output_dim_N=1,
        output_dim_P=1,
        output_dim_Si=1,
    ):
        super().__init__()
        self.feature_embeds = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_features)])
        self.trans = FeatureTransformer(embed_dim=embed_dim, output_dim=embed_dim_foundation)
        self.predictor_N = Predictor(embed_dim_foundation, output_dim_N)
        self.predictor_P = Predictor(embed_dim_foundation, output_dim_P)
        self.predictor_Si = Predictor(embed_dim_foundation, output_dim_Si)
        self.log_sigma_N = nn.Parameter(torch.zeros(1))
        self.log_sigma_P = nn.Parameter(torch.zeros(1))
        self.log_sigma_Si = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        tokens = []
        for i, embed in enumerate(self.feature_embeds):
            tokens.append(embed(x[:, i : i + 1]).unsqueeze(1))
        x = torch.cat(tokens, dim=1)
        x = self.trans(x)
        return self.predictor_N(x), self.predictor_P(x), self.predictor_Si(x)


def build_model(num_features, embed_dim=64, embed_dim_foundation=64):
    return FusionModel(
        num_features=num_features,
        embed_dim=embed_dim,
        embed_dim_foundation=embed_dim_foundation,
        output_dim_N=1,
        output_dim_P=1,
        output_dim_Si=1,
    )
