import torch
import torch.nn as nn
from ..base_layer import BaseLayer


class LinearLayer(BaseLayer):
    def __init__(self, in_features: int, out_features: int):
        self.torch_equiv_layer = nn.Linear(in_features, out_features)
        self.weight = torch.empty((out_features, in_features))
        self.bias = torch.empty(out_features)

    def forward(self, x):
        pass
