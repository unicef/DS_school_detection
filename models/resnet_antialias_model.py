# -*- coding: utf-8 -*-
import antialiased_cnns
import numpy as np
import torch
import torch.nn as nn


class ResNetModelAntiAlias(nn.Module):
    """
    Pre-trained ResNet18 Anti Alias Model trained on ImageNet.

    Parameters
    ----------

    n_classes: int
        Number of classes for classification

    pretrained: bool
        Download a pretrained imagenet model
    """

    hidden_size = 512

    def __init__(self, n_classes, pretrained=False, filter_size=4):
        super().__init__()
        model = antialiased_cnns.resnet18(pretrained, filter_size=filter_size)
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.hidden_size, n_classes)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            probabilities, classes = torch.softmax(self.model(x).cpu(), dim=1).topk(1)
            return classes.squeeze().numpy(), probabilities.squeeze().numpy()

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
