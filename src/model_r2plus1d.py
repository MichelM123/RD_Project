import torch
import torch.nn as nn

from torchvision.models.video import (
    r2plus1d_18,
    R2Plus1D_18_Weights,
)


class EpicR2Plus1DResNet(nn.Module):
    """
    EPIC-KITCHENS model using R(2+1)D-18 backbone (Kinetics-400 pretrained)
    with two classification heads:
        - verb: 97-way
        - noun: 300-way
    """

    def __init__(
        self,
        num_verbs: int = 97,
        num_nouns: int = 300,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Base backbone
        if pretrained:
            weights = R2Plus1D_18_Weights.KINETICS400_V1
        else:
            weights = None

        base = r2plus1d_18(weights=weights)

        # Remove final FC; keep backbone up to global pooling
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B, C, 1, 1, 1]
        self.feat_dim = base.fc.in_features  # usually 512

        self.dropout = nn.Dropout(p=dropout)

        # Two classification heads
        self.fc_verb = nn.Linear(self.feat_dim, num_verbs)
        self.fc_noun = nn.Linear(self.feat_dim, num_nouns)

        # Init heads (backbone already pretrained)
        nn.init.normal_(self.fc_verb.weight, 0, 0.01)
        nn.init.constant_(self.fc_verb.bias, 0)
        nn.init.normal_(self.fc_noun.weight, 0, 0.01)
        nn.init.constant_(self.fc_noun.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, T, H, W]
        Returns:
            verb_logits: [B, num_verbs]
            noun_logits: [B, num_nouns]
        """
        feats = self.backbone(x)               # [B, C, 1, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B, C]
        feats = self.dropout(feats)

        verb_logits = self.fc_verb(feats)
        noun_logits = self.fc_noun(feats)

        return verb_logits, noun_logits
