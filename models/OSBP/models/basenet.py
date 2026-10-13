# OSBP feature extractor, classifier and gradient-reversal utilities.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------- Gradient Reversal Layer ----------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# Feature extractor
class FeatureBase(nn.Module):
    """Transform an [N, feature_dim] array into learned feature vectors.

    Inputs are pre-extracted NPY features such as wav2vec, WavLM or ResNet.
    """

    def __init__(self, input_dim=2048, unit_size=512):
        super(FeatureBase, self).__init__()
        self.linear1 = nn.Linear(input_dim, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size)

    def forward(self, x, reverse=False):
        # Detach the input features when reverse is enabled.
        if reverse:
            x = x.detach()
        # Transform the pre-extracted features with fully connected layers.
        x = F.dropout(F.relu(self.bn1(self.linear1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.linear2(x))), training=self.training)
        return x


# Classifier
class ResClassifier(nn.Module):
    """Map FeatureBase outputs to class logits."""

    def __init__(self, num_classes=12, unit_size=512):
        super(ResClassifier, self).__init__()
        self.classifier = nn.Linear(unit_size, num_classes)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        logits = self.classifier(x)
        if return_feat:
            return logits, x
        return logits
