import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLayer(nn.Module):
    torch_equiv_layer: nn.Module

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def test(self, x) -> bool:
        result = self.forward(x)
        torch_result = self.torch_equiv_layer(x)
        return torch.allclose(result, torch_result)
