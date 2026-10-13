# SDEOTR feature-network components.

"""
models.py

A modular PyTorch implementation skeleton for SDEOTR:
Self-Supervised Domain Exploration with Optimal Transport Regularization
for open-set cross-domain speech emotion recognition.

This file contains:
    1. An SER backbone: G_f + G_y
    2. A SimSiam-style predictor for SSL-based domain exploration

The paper uses WavLM acoustic features and an ECAPA-TDNN emotion encoder.
Here, the encoder is implemented as a light MLP over pre-extracted features.
You can replace FeatureEncoder with an ECAPA-TDNN implementation if your
features are frame-level WavLM representations.
"""

from typing import Tuple

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Simplified embedding extractor G_f.

    Supported input shapes:
        - [B, D]      : utterance-level features
        - [B, T, D]   : frame-level features, averaged over time before MLP

    Args:
        input_dim: feature dimension D, e.g. 768 for WavLM-base.
        embed_dim: output embedding dimension, 512 in the paper.
        dropout: dropout probability.
    """

    def __init__(
        self, input_dim: int = 768, embed_dim: int = 512, dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x is frame-level [B, T, D], aggregate to utterance-level [B, D].
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


class EmotionClassifier(nn.Module):
    """Emotion classifier G_y over known source classes."""

    def __init__(self, embed_dim: int = 512, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.fc(e)


class SDEOTRModel(nn.Module):
    """
    Main SER network f(x)=G_y(G_f(x)).

    During training, this backbone is shared by:
        - source supervised classification
        - target SimSiam SSL branch
        - target OTR branch

    During inference, only this backbone is retained.
    """

    def __init__(
        self,
        input_dim: int = 768,
        embed_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            input_dim=input_dim, embed_dim=embed_dim, dropout=dropout
        )
        self.classifier = EmotionClassifier(
            embed_dim=embed_dim, num_classes=num_classes
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classify(self, e: torch.Tensor) -> torch.Tensor:
        return self.classifier(e)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.encode(x)
        logits = self.classify(e)
        return logits, e


class SimSiamPredictor(nn.Module):
    """
    Predictor head used in the SSL-based domain exploration module.

    The paper describes a SimSiam-style predictor with two FC layers and ReLU.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)
