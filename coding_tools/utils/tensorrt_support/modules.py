import torch
from torch import Tensor


class TorchTensorRTPlaceholder(torch.nn.Identity):
    def forward(self, input: Tensor) -> Tensor:
        raise ValueError("Placeholder cannot be forwarded.")
