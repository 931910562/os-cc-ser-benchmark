# Feature classifiers and adversarial networks for ANNA.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# Gradient Reversal Layer
# ================================
class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


class RevGradLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)


def grad_reverse(x, lambd=1.0):
    return RevGradLayer(lambd)(x)


# ================================
# Feature-based network
# ================================
class Net(nn.Module):
    """Classify pre-extracted features with shape [B, D] without a CNN backbone."""

    def __init__(self, args):
        super(Net, self).__init__()
        self.num_class = args.known_class + 1
        self.feature_dim = args.feature_dim  # Input feature dimension.

        # Use pre-extracted features directly instead of a CNN backbone.
        self.generator = Identity()
        self.classifier = Classifier(
            self.num_class, in_dim=self.feature_dim, unit_size=1024
        )

        dim = 1024 if args.all_layer_adv else self.feature_dim
        self.adv_k = AdversarialNetwork(dim)
        self.adv_unk = AdversarialNetwork(dim)

    def forward(self, x, constant=1, adaption=False):
        feats = self.generator(x)  # Identity mapping.
        out = self.classifier(feats, constant, adaption)
        return out


# ================================
# Identity feature generator
# ================================
class Identity(nn.Module):
    def forward(self, x):
        return x


# ================================
# Classifier
# ================================
class Classifier(nn.Module):
    def __init__(self, num_classes, in_dim=1024, unit_size=100):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(in_dim, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.classifier = nn.Linear(unit_size, num_classes)
        self.drop = nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rois, constant=1, adaption=False, return_feat=False):
        x = self.drop(F.relu(self.bn1(self.linear1(rois))))
        x = self.drop(F.relu(self.bn2(self.linear2(x))))
        x_rev = grad_reverse(x, constant) if adaption else x
        logits = self.classifier(x_rev)
        return (logits, x) if return_feat else logits


# ================================
# Adversarial network
# ================================
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main1 = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant=0.05):
        x = grad_reverse(x, constant)
        return self.main1(x).view(-1)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
