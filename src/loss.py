import abc
from ast import Dict
from math import log10

import torch

from .fileio import FileIO
from typing import List, Type, Union

import numpy as np
import pytorch_msssim


class LossBase(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def ctu_level_loss(
        origin: "ImageBlock", recon: Union[np.ndarray, torch.Tensor]
    ) -> float:
        pass

    @staticmethod
    @abc.abstractmethod
    def global_level_loss(fileio: FileIO, ctu_level_losses: List[float]) -> float:
        pass


LOSSES: dict[str, Type[LossBase]] = {}


def _register_loss(name: str):
    def _f(losscls: Type[LossBase]):
        LOSSES.setdefault(name, losscls)
        return losscls

    return _f


@_register_loss("PSNR")
class PSNRLoss(LossBase):
    @staticmethod
    def ctu_level_loss(
        origin: "ImageBlock", recon: Union[np.ndarray, torch.Tensor]
    ) -> float:
        if isinstance(recon, torch.Tensor):
            sqe = torch.sum((origin.cuda - recon) ** 2).detach().cpu().numpy()
        else:
            sqe = np.sum((origin.np - recon) ** 2) / (255.0**2)
        return sqe

    @staticmethod
    def global_level_loss(fileio: FileIO, ctu_level_losses: List[float]) -> float:
        num_pixels = fileio.num_pixels
        mse = sum(ctu_level_losses) / num_pixels / (255.0**2)
        negpsnr = -10 * log10(mse)
        return negpsnr


@_register_loss("MS-SSIM")
class MSSSIMLoss(LossBase):
    @staticmethod
    def ctu_level_loss(
        origin: "ImageBlock", recon: Union[np.ndarray, torch.Tensor]
    ) -> float:
        X: torch.Tensor = origin.cuda
        if isinstance(recon, torch.Tensor):
            Y = recon
        else:
            Y = torch.tensor(recon)
            Y = Y.permute((2, 0, 1)).to(torch.float32) / 255.0
            Y = Y.unsqueeze(0)
        Y = Y.to(X.device)
        msssim = pytorch_msssim.ms_ssim(X, Y, data_range=1.0).detach().cpu().numpy()
        _, c, h, w = X.shape
        ctu_loss = h * w * (1.0 - msssim)
        return ctu_loss

    @staticmethod
    def global_level_loss(fileio: FileIO, ctu_level_losses: List[float]) -> float:
        tot_loss = sum(ctu_level_losses) / fileio.num_pixels
        negmsssim = tot_loss - 1
        return negmsssim
