"""
Python implementation of boundary-aware MS-SSIM
"""

from typing import Optional, Tuple, Union, List

from pytorch_msssim.ssim import *
from pytorch_msssim.ssim import _fspecial_gauss_1d
import numpy as np
from dataclasses import dataclass


@dataclass
class BoundaryInfo:
    top: bool = False
    left: bool = False
    bottom: bool = False
    right: bool = False

    @property
    def padding_mask(self) -> np.ndarray:
        return np.asarray(
            [self.left, self.right, self.top, self.bottom], dtype=np.int32
        )

    def get_padding(self, pd_size):
        pd = self.padding_mask * (pd_size)
        pd = tuple(pd)
        return pd


def ba_gaussian_filter(input: Tensor, win: Tensor, boundary: BoundaryInfo) -> Tensor:
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]

    # Pad image
    pd_size = win.shape[3] // 4
    out = F.pad(input=input, pad=boundary.get_padding(pd_size), mode="reflect")

    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(
                out, weight=win.transpose(2 + i, -1)[:C], stride=1, padding=0, groups=C
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ba_ssim(
    X: Tensor,
    Y: Tensor,
    boundary: BoundaryInfo,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
) -> Tuple[Tensor, Tensor]:
    r"""Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win_size = win.shape[3]
    win = win.to(X.device, dtype=X.dtype)

    mu1 = ba_gaussian_filter(X, win, boundary)
    mu2 = ba_gaussian_filter(Y, win, boundary)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (ba_gaussian_filter(X * X, win, boundary) - mu1_sq)
    sigma2_sq = compensation * (ba_gaussian_filter(Y * Y, win, boundary) - mu2_sq)
    sigma12 = compensation * (ba_gaussian_filter(X * Y, win, boundary) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    valid_pixels = (
        torch.ones(X.shape[2:]).unsqueeze(0).unsqueeze(0).to(X.device, dtype=X.dtype)
    )
    pd_size = win_size // 4
    valid_pixels = F.pad(
        valid_pixels, pad=boundary.get_padding(pd_size), mode="constant", value=0
    )
    weight_kernel = torch.ones_like(win)
    weight_map = ba_gaussian_filter(valid_pixels, weight_kernel, BoundaryInfo())
    weight_map = weight_map / weight_map.sum()

    ssim_per_channel = torch.sum(ssim_map * weight_map, axis=[2, 3])
    cs = torch.sum(cs_map * weight_map, axis=[2, 3])

    return ssim_per_channel, cs


def ba_ssim(
    X: Tensor,
    Y: Tensor,
    boundary: BoundaryInfo,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(
            f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}."
        )

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    # if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ba_ssim(
        X, Y, boundary, data_range=data_range, win=win, size_average=False, K=K
    )
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ba_ms_ssim(
    X: Tensor,
    Y: Tensor,
    boundary: BoundaryInfo,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
) -> Tensor:
    r"""interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(
            f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}."
        )

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    # if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2**4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % (
        (win_size - 1) * (2**4)
    )

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ba_ssim(
            X, Y, boundary, win=win, data_range=data_range, size_average=False, K=K
        )

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(
        mcs + [ssim_per_channel], dim=0
    )  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)
