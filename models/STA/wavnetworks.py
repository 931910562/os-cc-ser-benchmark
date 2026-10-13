# Feature-based networks for the two-stage STA experiment.

import torch
import torch.nn as nn
from torch.autograd import Function


# -----------------------------
# Gradient Reversal Layer
# -----------------------------
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return None, -ctx.coeff * grad_output


class GradientReverseModule(nn.Module):
    def __init__(self, scheduler=None):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler if scheduler is not None else (lambda step: 1.0)
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)


# -----------------------------
# Wav2Vec Feature Extractor
# -----------------------------
class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, input_dim=768):
        super(Wav2VecFeatureExtractor, self).__init__()
        # Store the input dimension for the identity feature extractor.
        self.input_dim = input_dim

    def forward(self, x):
        # Return the pre-extracted feature batch unchanged.
        return x

    def output_num(self):
        return self.input_dim


# -----------------------------
# CLS (Classifier)
# -----------------------------
class CLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        feature = x  # Input features
        logits = self.fc(x)  # Class logits
        probs = self.softmax(logits)  # Softmax probabilities
        return x, feature, logits, probs


# -----------------------------
# CLS_0 (for discriminator_t)
# -----------------------------
class CLS_0(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS_0, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_feature = x
        logits = self.fc(x)
        probs = self.softmax(logits)
        # Return four values for compatibility with the training interface.
        return input_feature, input_feature, logits, probs


# -----------------------------
# Adversarial Network
# -----------------------------
class AdversarialNetwork(nn.Module):
    def __init__(self, grl_scheduler=None):
        super(AdversarialNetwork, self).__init__()
        self.grl = GradientReverseModule(grl_scheduler)
        self.main = nn.Sequential()

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x


class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid,
        )


# -----------------------------
# Multi-binary Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, n=10, input_dim=768):
        super(Discriminator, self).__init__()
        self.n = n

        def f():
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)
