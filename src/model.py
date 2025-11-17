# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50


class Epic2DResNet(nn.Module):
    """
    TSN-style 2D ResNet for EPIC-KITCHENS.

    Input:  clips [B, C, T, H, W]  (C=3)
    Steps:
      - permute to [B, T, C, H, W]
      - merge batch & time -> [B*T, C, H, W]
      - ResNet-50 backbone
      - temporal average over T
      - two classification heads: verb, noun
    """
    def __init__(self, num_verbs: int, num_nouns: int, pretrained: bool = True):
        super().__init__()

        if pretrained:
            base = resnet50(weights="IMAGENET1K_V2")
        else:
            base = resnet50(weights=None)

        # Backbone up to global average pooling
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.feat_dim = base.fc.in_features  # 2048 for ResNet-50

        # Heads
        self.fc_verb = nn.Linear(self.feat_dim, num_verbs)
        self.fc_noun = nn.Linear(self.feat_dim, num_nouns)

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)          # [B*T, feat_dim, 1, 1]
        feats = feats.view(B, T, -1)      # [B, T, feat_dim]

        vid_feat = feats.mean(dim=1)      # temporal average -> [B, feat_dim]

        verb_logits = self.fc_verb(vid_feat)
        noun_logits = self.fc_noun(vid_feat)
        return verb_logits, noun_logits
