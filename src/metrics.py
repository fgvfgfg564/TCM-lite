import abc
from numpy.core.multiarray import array as array
import torch
import numpy as np
import pytorch_msssim

from fileio import FileIO


def torch_to_uint8(x):
    x = torch.clamp(x, 0, 1)
    x *= 255
    x = torch.round(x)
    x = x.to(torch.uint8)
    return x


def convert_to_np_uint8(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = torch_to_uint8(x)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()
    return x


class MetricBase(abc.ABC):
    @classmethod
    def ctu_level_loss(
        cls, output: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor
    ) -> float:
        output = convert_to_np_uint8(output)
        target = convert_to_np_uint8(target)
        return cls._ctu_level_loss_np(output, target)

    @classmethod
    @abc.abstractmethod
    def _ctu_level_loss_np(cls, output: np.ndarray, target: np.ndarray) -> float:
        """
        loss on CTU level. I/O in numpy uint8 format
        """

    @classmethod
    @abc.abstractmethod
    def global_level_loss(cls, ctu_level_losses: np.ndarray, file_io: FileIO):
        """
        Global loss reducer.
        """


class PSNRMetric(MetricBase):
    @classmethod
    def _ctu_level_loss_np(cls, output: np.ndarray, target: np.ndarray) -> float:
        output = output.astype(np.float32)
        target = target.astype(np.float32)
        return np.sum((output - target) ** 2) / (255.0**2)

    @classmethod
    def global_level_loss(cls, ctu_level_losses: np.ndarray, file_io: FileIO):
        sqe_sum = ctu_level_losses.sum()
        num_pixels = file_io.num_pixels
        mse = sqe_sum / num_pixels * 3
        psnr = -10 * np.log10(mse)
        return psnr


class MSSSIMMEtric(MetricBase):
    @classmethod
    def _ctu_level_loss_np(cls, output: np.ndarray, target: np.ndarray) -> float:
        output = torch.tensor(output).permute(2, 0, 1).unsqueeze(0)
        target = torch.tensor(target).permute(2, 0, 1).unsqueeze(0)
        return pytorch_msssim.ms_ssim(output, target, data_range=255).numpy().item()

    @classmethod
    def global_level_loss(cls, ctu_level_losses: np.ndarray, file_io: FileIO):
        # Calculate the number of MS-SSIM windows according to the CTU size.
        w = file_io.block_num_pixels

        return ctu_level_losses
