# STIENet network components.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Encoder for spectrogram input
# ============================================================
class STIENetEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, kernel_size=(3, 3), padding=1
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=(3, 3), padding=1
        )
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.conv3 = nn.Conv2d(
            base_channels * 2, base_channels * 4, kernel_size=(3, 3), padding=1
        )
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        # flatten frequency/time dims -> sequence
        B, C, H, W = x.size()
        seq = x.view(B, C * H, W).transpose(1, 2)  # [B, T', feat_dim]
        return seq


# ============================================================
# Full STIENet model
# ============================================================
class STIENet(nn.Module):
    def __init__(
        self, n_mels=64, base_channels=64, lstm_hidden=64, lstm_layers=2, num_known=3
    ):
        super().__init__()
        self.num_known = num_known
        self.is_resnet_input = n_mels == 1024  # detect ResNet features

        if not self.is_resnet_input:
            # Standard audio-based encoder
            self.encoder = STIENetEncoder(in_channels=1, base_channels=base_channels)
            lstm_input_dim = base_channels * 4 * (n_mels // 8)
            self.lstm = nn.LSTM(
                lstm_input_dim,
                lstm_hidden,
                lstm_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.Linear(lstm_hidden * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_known + 1),
            )
        else:
            # Direct ResNet feature mode
            self.feature_proj = nn.Sequential(
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3)
            )
            self.classifier = nn.Linear(512, num_known + 1)

    def forward(self, x):
        """
        Forward pass.
        - If x is spectrogram: shape [B, 1, n_mels, T]
        - If x is ResNet features: shape [B, 2048]
        """
        if self.is_resnet_input:
            # ---- ResNet feature mode ----
            feats = self.feature_proj(x)
            logits = self.classifier(feats)
            return logits, feats
        else:
            # ---- Spectrogram mode ----
            seq = self.encoder(x)  # [B, T', feat_in]
            out, (h, _) = self.lstm(seq)  # [B, T', 2*lstm_hidden]
            h_out = torch.cat((h[-2], h[-1]), dim=1)  # [B, 2*lstm_hidden]
            logits = self.classifier(h_out)
            return logits, h_out
