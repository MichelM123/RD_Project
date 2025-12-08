import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

try:
    from torchvision.models.video import R3D_18_Weights
    HAS_WEIGHTS_ENUM = True
except ImportError:
    HAS_WEIGHTS_ENUM = False


class Epic3DResNet(nn.Module):
    """
    3D ResNet-18 backbone on video clips [B, C, T, H, W],
    with two classification heads:
        - verbs  (num_verbs classes)
        - nouns  (num_nouns classes)

    Includes a dropout layer before the heads for regularization.
    """

    def __init__(
        self,
        num_verbs: int,
        num_nouns: int,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        if pretrained:
            if HAS_WEIGHTS_ENUM:
                base = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            else:
                base = r3d_18(pretrained=True)
        else:
            if HAS_WEIGHTS_ENUM:
                base = r3d_18(weights=None)
            else:
                base = r3d_18(pretrained=False)

        # r3d_18 output: [B, 512, 1, 1, 1] after global pooling
        # Strip the final FC layer and keep everything up to global pool
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.feat_dim = base.fc.in_features  # 512 for r3d_18

        self.dropout = nn.Dropout(p=dropout)

        # Heads
        self.fc_verb = nn.Linear(self.feat_dim, num_verbs)
        self.fc_noun = nn.Linear(self.feat_dim, num_nouns)

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, T, H, W]
        returns:
            verb_logits: [B, num_verbs]
            noun_logits: [B, num_nouns]
        """
        # r3d_18 already expects [B, C, T, H, W]
        feats = self.backbone(x)  # [B, feat_dim, 1, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B, feat_dim]
        feats = self.dropout(feats)

        verb_logits = self.fc_verb(feats)
        noun_logits = self.fc_noun(feats)
        return verb_logits, noun_logits
