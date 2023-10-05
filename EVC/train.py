import argparse
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder

from src.models import build_model, model_architectures
from torch.utils.tensorboard import SummaryWriter
import os

from dataset import VCIP_Training

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = torch.mean((output["x_hat"] - target)**2,dim=(1,2,3))
        out["dloss"] = 255 ** 2 * torch.dot(lmbda, out["mse_loss"])
        out["mse_loss"] = torch.mean(out["mse_loss"])
        out["loss"] = out["dloss"] + out["bpp_loss"]
        out["psnr_loss"] = -10*torch.log10(out["mse_loss"])

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net: torch.nn.Module, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = net.parameters()
    optimizer = optim.Adam(
        parameters,
        lr=args.learning_rate,
    )
    return optimizer


def train_one_epoch(
    model, lmbdas, criterion, train_dataloader, optimizer, epoch, clip_max_norm, type='mse'
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()

        # randomly select lmbda
        B, _, H, W = d.shape
        lmbda_index = np.random.randint(0, len(lmbdas), size=[B])
        lmbda = lmbdas[lmbda_index]
        qscale = torch.gather(model.q_scale, 0, lmbda_index)

        out_net = model(d, qscale)

        out_criterion = criterion(out_net, d, lmbda)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()

        if i % 1000 == 0:
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


def test_epoch(epoch, test_dataloader, lmbdas, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        losses = []
        for (i, lmd) in lmbdas:
            loss = AverageMeter()
            bpp_loss = AverageMeter()
            mse_loss = AverageMeter()
            psnr_loss = AverageMeter()

            for d in test_dataloader:
                d = d.to(device)
                qscale = model.q_scale[i]
                out_net = model(d, qscale)
                out_criterion = criterion(out_net, d)

                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                psnr_loss.update(out_criterion["psnr_loss"])

            print(f"Testing results for lmbda={lmd:.4f}: loss={loss.avg:.5f} - bpp_loss={bpp_loss.avg:.5f} - mse_loss={mse_loss.avg:.5f} - avg_psnr={psnr_loss:.3f}")
            losses.append(loss)

    return np.mean(loss)


def save_checkpoint(state, is_best, save_path):
    torch.save(state, save_path + "last_epoch.pth.tar")

    if is_best:
        torch.save(state, save_path + "best_epoch.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("model", type=str, choices=list(model_architectures.keys()))
    parser.add_argument("-l", "--lmbdas", nargs=4, type=float, help="lambdas for training", required=True)
    parser.add_argument(
        "-e",
        "--epochs",
        default=1000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--batch_size", "-bs", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=19260817, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, default='./', help="save_path")
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int, default=[10]
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path)
    tb_path = os.path.join(save_path, "tensorboard/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(tb_path)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(tb_path)
    
    train_dataset = VCIP_Training(patch_size=args.patch_size)
    test_dataset = VCIP_Training(patch_size=args.patch_size, buffer_size=128, stable=True)

    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = build_model(args.model)
    net = net.to(device)

    optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss()

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
    lmbdas = args.lmbdas

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            lmbdas,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            type
        )
        loss = test_epoch(epoch, test_dataloader, lmbdas, net, criterion)
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                save_path,
            )


if __name__ == "__main__":
    main(sys.argv[1:])