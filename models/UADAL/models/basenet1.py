# UADAL feature-network and classifier components.

import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torch.autograd.variable import *

# from efficientnet_pytorch import EfficientNet
import math


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass


## Some classes from https://github.com/ksaito-ut/OPDA_BP/blob/master/models/basenet.py

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


class ResNetFc(nn.Module):
    """
    Feature extractor for pre-extracted features (e.g., from .npy files).
    Does not use ResNet anymore; optionally applies a simple linear transformation
    or directly returns the input features.
    """

    def __init__(self, model_name='feature', input_dim=88, normalize=False):
        """
        Args:
            model_name (str): Placeholder for compatibility (unused here)
            input_dim (int): Dimension of input feature vector
            normalize (bool): Whether to normalize input features along the feature dimension
        """
        super(ResNetFc, self).__init__()
        self.normalize = normalize
        self.input_dim = input_dim
        self.__in_features = input_dim  # Output dimension matches input dimension

        # Optional lightweight transformation layer
        # Use Identity to return input directly, or Linear for projection
        # self.projector = nn.Identity()
        self.projector = nn.Linear(input_dim, input_dim)

        # Example of linear projection: nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, input_dim]
        Returns:
            torch.Tensor: Output features of shape [batch_size, input_dim]
        """
        if self.normalize:
            x = F.normalize(x, dim=1)  # Normalize features along feature dimension
        x = self.projector(x)
        return x

    def output_num(self):
        """
        Returns:
            int: Dimension of output features
        """
        return self.__in_features


class DenseNet(nn.Module):
    """
    Feature extractor for pre-extracted features (e.g., from .npy files).
    Does not use ResNet anymore; optionally applies a simple linear transformation
    or directly returns the input features.
    """

    def __init__(self, model_name='feature', input_dim=1024, normalize=False):
        """
        Args:
            model_name (str): Placeholder for compatibility (unused here)
            input_dim (int): Dimension of input feature vector
            normalize (bool): Whether to normalize input features along the feature dimension
        """
        super(DenseNet, self).__init__()
        self.normalize = normalize
        self.input_dim = input_dim
        self.__in_features = input_dim  # Output dimension matches input dimension

        # Optional lightweight transformation layer
        # Use Identity to return input directly, or Linear for projection
        # self.projector = nn.Identity()
        self.projector = nn.Linear(input_dim, input_dim)

        # Example of linear projection: nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, input_dim]
        Returns:
            torch.Tensor: Output features of shape [batch_size, input_dim]
        """
        if self.normalize:
            x = F.normalize(x, dim=1)  # Normalize features along feature dimension
        x = self.projector(x)
        return x

    def output_num(self):
        """
        Returns:
            int: Dimension of output features
        """
        return self.__in_features


class Net_CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Net_CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        return x


class Net_CLS_C(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim, bias=True):
        super(Net_CLS_C, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=bias)
        self.main = nn.Sequential(
            self.bottleneck,
            nn.Sequential(
                nn.BatchNorm1d(bottle_neck_dim),
                nn.LeakyReLU(0.2, inplace=True),
                self.fc,
            ),
        )

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x


class Net_CLS_DC(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=None):
        super(Net_CLS_DC, self).__init__()
        if bottle_neck_dim is None:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
            )
        else:
            self.main = nn.Sequential(
                nn.Linear(in_dim, bottle_neck_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(bottle_neck_dim, bottle_neck_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(bottle_neck_dim, out_dim),
            )

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [
            torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)
        ]

    def forward(self, input_list):
        return_list = [
            torch.mm(input_list[i], self.random_matrix[i])
            for i in range(self.input_num)
        ]
        return_tensor = return_list[0] / math.pow(
            float(self.output_dim), 1.0 / len(return_list)
        )
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        return x

    def output_num(self):
        return 100
