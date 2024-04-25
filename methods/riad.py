import math
import random
from statistics import mean
from typing import Dict, List, Tuple
import numpy as np
from numpy import ndarray as NDArray
from torch import Tensor
import os
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

import utils
from methods.unsupervised_method import UnsupervisedMethod
import torch.nn.functional as F
from config import Config
from torch.nn import Module
from utils import plot_sample


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: int = 0) -> None:

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = np.Inf

    def __call__(self, score: float) -> bool:

        if self.best_score is None:
            self.best_score = score
            return False

        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False

        else:
            self.best_score = score
            self.counter = 0
            return False


class MSGMSLoss(Module):
    def __init__(self, num_scales: int = 3, in_channels: int = 3) -> None:

        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.prewitt_x, self.prewitt_y = self._create_prewitt_kernel()

    def forward(self, img1: Tensor, img2: Tensor, as_loss: bool = True) -> Tensor:

        if not self.prewitt_x.is_cuda or not self.prewitt_y.is_cuda:
            self.prewitt_x = self.prewitt_x.to(img1.device)
            self.prewitt_y = self.prewitt_y.to(img1.device)

        b, c, h, w = img1.shape
        msgms_map = 0
        for scale in range(self.num_scales):

            if scale > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)

            gms_map = self._gms(img1, img2)
            msgms_map += F.interpolate(gms_map, size=(h, w), mode="bilinear", align_corners=False)

        if as_loss:
            return torch.mean(1 - msgms_map / self.num_scales)
        else:
            return torch.mean(1 - msgms_map / self.num_scales, axis=1).unsqueeze(1)

    def _gms(self, img1: Tensor, img2: Tensor) -> Tensor:

        gm1_x = F.conv2d(img1, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm1_y = F.conv2d(img1, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm1 = torch.sqrt(gm1_x ** 2 + gm1_y ** 2 + 1e-12)

        gm2_x = F.conv2d(img2, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm2_y = F.conv2d(img2, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm2 = torch.sqrt(gm2_x ** 2 + gm2_y ** 2 + 1e-12)

        # Constant c from the following paper. https://arxiv.org/pdf/1308.3052.pdf
        c = 0.0026
        numerator = 2 * gm1 * gm2 + c
        denominator = gm1 ** 2 + gm2 ** 2 + c
        return numerator / (denominator + 1e-12)

    def _create_prewitt_kernel(self) -> Tuple[Tensor, Tensor]:

        prewitt_x = torch.Tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_x = prewitt_x.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        prewitt_y = torch.Tensor([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_y = prewitt_y.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        return (prewitt_x, prewitt_y)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode="concat", up_mode="transpose"):
        super(UNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 4, 2, 1)
        self.down3 = UNetDownBlock(128, 256, 4, 2, 1)
        self.down4 = UNetDownBlock(256, 512, 4, 2, 1)
        self.down5 = UNetDownBlock(512, 512, 4, 2, 1)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)

        self.conv_final = nn.Sequential(conv(64, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x


class SSIMLoss(Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:

        """Computes the structural similarity (SSIM) index map between two images

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:

        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:

        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:

        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d


def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


class Riad(UnsupervisedMethod):

    def train_and_eval(self, curr_run_path, cfg: Config, train_dataloader, test_dataloader, extra_args):
        self.c = cfg
        self.device = cfg.DEVICE

        self.model = self._init_model().to(self.device)
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterions = {k: self._init_criterions(k) for k in ["MSGMS", "SSIM", "MSE"]}
        self.early_stopping = self._init_early_stopping()

        for epoch in range(1, self.c.N_EPOCHS + 1):
            train_metrics = self._train(epoch, train_dataloader)
            # val_metrics = self._validate(epoch, test_dataloader)
            self.scheduler.step()

            if self.early_stopping(mean(train_metrics["Total"])):
                print(f"Early stopped at {epoch} epoch")
                break

            # if epoch % 5 == 0:
            #     ts = self._test(test_dataloader)
            #     tr = self._test(train_dataloader)
            #     test_auc, train_auc = get_auc(ts["gt"], ts["ascore"]), get_auc(tr["gt"], tr["ascore"])
            #     print(f"TEST: Epoch {epoch + 1}, TRAIN_AUC:{train_auc}, TEST_AUC:{test_auc}")

        test_data = self._test(test_dataloader, cfg.SAVE_SEGMENTATION, os.path.join(curr_run_path, "segmentation"))
        train_data = self._test(train_dataloader, False, None)

        torch.save(self.model.state_dict(), os.path.join(curr_run_path, f"model_FINAL.pth"))

        return_dict = {
            "tr_gts": train_data["gt"], "tr_scores": train_data["ascore"], "tr_names": train_data["name"],
            "ts_gts": test_data["gt"], "ts_scores": test_data["ascore"], "ts_names": test_data["name"],
        }
        return return_dict

    def _init_model(self) -> Module:
        model = UNet()
        return model

    def _init_criterions(self, key: str) -> Module:
        if key == "MSGMS":
            return MSGMSLoss(num_scales=4)
        elif key == "SSIM":
            return SSIMLoss()
        else:
            return torch.nn.MSELoss()

    def _init_optimizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), self.c.LEARNING_RATE, weight_decay=self.c.RIAD_WEIGHT_DECAY)
        # return torch.optim.SGD(self.model.parameters(), self.c.LEARNING_RATE, weight_decay=self.c.WEIGHT_DECAY, momentum=0.9, nesterov=True)

    def _init_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10, 2, 1e-4, -1)
        # return torch.optim.lr_scheduler.StepLR(self.optimizer, self.c.LR_STEP, self.c.LR_GAMMA)

    def _init_early_stopping(self) -> EarlyStopping:
        es = EarlyStopping(10, 0)
        return es

    def report_metrics(self, metrics, kind, epoch):
        metrics_str = ", ".join([f"{k}:{mean(v):.3f}" for k, v in metrics.items()])
        print(f"{kind.upper()} {epoch}||| {metrics_str}")

    def _train(self, epoch, train_dataloader) -> Dict[str, List[float]]:

        metrics: Dict[str, List[float]] = {
            "MSE": [],
            "MSGMS": [],
            "SSIM": [],
            "Total": [],
        }
        self.model.train()
        for sample_dict in train_dataloader:
            mb_img = sample_dict["image"]
            self.optimizer.zero_grad()

            mb_img = mb_img.to(self.device)
            cutout_size = random.choice(self.c.RIAD_CUTOUT_SIZES)
            mb_reconst = self._reconstruct(mb_img, cutout_size)

            mb_mse = self.criterions["MSE"](mb_img, mb_reconst)
            mb_msgms = self.criterions["MSGMS"](mb_img, mb_reconst)
            mb_ssim = self.criterions["SSIM"](mb_img, mb_reconst)
            mb_total = mb_msgms + mb_ssim + mb_mse
            mb_total.backward()
            self.optimizer.step()

            metrics["MSE"].append(mb_mse.item())
            metrics["MSGMS"].append(mb_msgms.item())
            metrics["SSIM"].append(mb_ssim.item())
            metrics["Total"].append(mb_total.item())

        self.report_metrics(metrics, "TRAIN", epoch)
        return metrics

    def _test(self, test_dataloader, save_segmentation, segmentation_save_dir):

        self.model.eval()
        artifacts: Dict[str, List[NDArray]] = {
            "img": [],
            "reconst": [],
            "gt": [],
            "amap": [],
            "name": [],
            "ascore": []
        }
        for sample_dict in test_dataloader:
            names = sample_dict["name"]
            mb_img = sample_dict["image"]
            mb_gt = sample_dict["label"]
            mb_amap = 0
            with torch.no_grad():
                for cutout_size in self.c.RIAD_CUTOUT_SIZES:
                    mb_img = mb_img.to(self.device)
                    mb_reconst = self._reconstruct(mb_img, cutout_size)
                    mb_amap += self.criterions["MSGMS"](mb_img, mb_reconst, as_loss=False)

            mb_amap = mean_smoothing(mb_amap)
            artifacts["amap"].extend(mb_amap.squeeze(1).detach().cpu().numpy())
            artifacts["img"].extend(mb_img.permute(0, 2, 3, 1).detach().cpu().numpy())
            artifacts["reconst"].extend(mb_reconst.permute(0, 2, 3, 1).detach().cpu().numpy())
            artifacts["gt"].extend(mb_gt.detach().cpu().numpy())
            artifacts["name"].extend(names)

        ep_amap = np.array(artifacts["amap"])
        ep_amap = (ep_amap - ep_amap.min()) / (ep_amap.max() - ep_amap.min())
        artifacts["amap"] = list(ep_amap)
        artifacts["ascore"] = np.array(list(map(np.max, ep_amap)))
        artifacts["gt"] = np.array(artifacts["gt"])
        artifacts["name"] = np.array(artifacts["name"])

        if save_segmentation:
            for img, seg, gt, score, name in zip(test_dataloader.dataset, artifacts["amap"], artifacts["gt"], artifacts["ascore"], artifacts["name"]):
                utils.plot_sample(img["image"].detach().cpu().numpy().transpose((1, 2, 0)), None, seg, gt.item(), name, score, segmentation_save_dir, normalize_imagenet=False, normalize_1=True)

        return artifacts

    def _reconstruct(self, mb_img: Tensor, cutout_size: int) -> Tensor:

        _, _, h, w = mb_img.shape
        num_disjoint_masks = self.c.RIAD_NUM_DISJOINT_MASKS
        disjoint_masks = self._create_disjoint_masks((h, w), cutout_size, num_disjoint_masks)

        mb_reconst = 0
        for mask in disjoint_masks:
            mb_cutout = mb_img * mask
            mb_inpaint = self.model(mb_cutout)
            mb_reconst += mb_inpaint * (1 - mask)

        return mb_reconst

    def _create_disjoint_masks(self, img_size: Tuple[int, int], cutout_size: int = 8, num_disjoint_masks: int = 3, ) -> List[Tensor]:
        img_h, img_w = img_size
        grid_h = math.ceil(img_h / cutout_size)
        grid_w = math.ceil(img_w / cutout_size)
        num_grids = grid_h * grid_w
        disjoint_masks = []
        for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
            flatten_mask = np.ones(num_grids)
            flatten_mask[grid_ids] = 0
            mask = flatten_mask.reshape((grid_h, grid_w))
            mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
            mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
            mask = mask.to(self.device)
            disjoint_masks.append(mask)

        return disjoint_masks
