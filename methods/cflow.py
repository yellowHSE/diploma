import torch.nn.functional as F
from tqdm import tqdm

from methods.unsupervised_method import UnsupervisedMethod
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from config import Config
import numpy as np
import os
import math
import torch
from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from utils import plot_sample

coupling_blocks = 8
clamp_alpha = 1.9
condition_vec = 128

model_urls = {
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
}

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

PADDING_MODE = 'reflect'  # {'zeros', 'reflect', 'replicate', 'circular'}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=PADDING_MODE, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, padding_mode=PADDING_MODE,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # remove extra layers
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # model.load_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))


def freia_flow_head(n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NF coder:', n_feat)
    for k in range(coupling_blocks):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=clamp_alpha,
                     global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_cflow_head(n_feat):
    n_cond = condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(coupling_blocks):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=clamp_alpha,
                     global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_decoder_arch(dec_arch, dim_in):
    if dec_arch == 'freia-flow':
        decoder = freia_flow_head(dim_in)
    elif dec_arch == 'freia-cflow':
        decoder = freia_cflow_head(dim_in)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(dec_arch))
    return decoder


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def load_encoder_arch(enc_arch, L):
    # encoder pretrained on natural images:
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ['layer' + str(i) for i in range(L)]

    if 'resnet' in enc_arch:
        if enc_arch == 'wide_resnet50_2':
            encoder = wide_resnet50_2(pretrained=True, progress=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(enc_arch))
        #
        if L >= 3:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in enc_arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in enc_arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in enc_arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1

    return encoder, pool_layers, pool_dims


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5 * torch.sum(z ** 2, 1) + logdet_J
    return logp


gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


class CFLOW(UnsupervisedMethod):
    n_pool_layers = 1

    enc_arch = "wide_resnet50_2"
    dec_arch = "freia-cflow"
    clamp_alpha = 1.9
    dropout = 0.0
    lr_warm = True
    lr_cosine = True
    lr_decay_rate = 0.1
    lr_warm_epochs = 2
    
    sub_epochs = 8

    def train_and_eval(self, curr_run_path, cfg: Config, train_dataloader, test_dataloader, extra_args):
        self.cfg: Config = cfg
        self.img_size = train_dataloader.dataset[0]["image"].shape[-2:]
        self.crp_size = train_dataloader.dataset[0]["image"].shape[-2:]
        self.lr_decay_epochs = [i * cfg.N_EPOCHS // 100 for i in [50, 75, 90]]
        if self.lr_warm:
            self.lr_warmup_from = cfg.LEARNING_RATE / 10.0
            if self.lr_cosine:
                eta_min = cfg.LEARNING_RATE * (self.lr_decay_rate ** 3)
                self.lr_warmup_to = eta_min + (cfg.LEARNING_RATE - eta_min) * (
                        1 + math.cos(math.pi * self.lr_warm_epochs / cfg.N_EPOCHS)) / 2
            else:
                self.lr_warmup_to = cfg.LEARNING_RATE

        device = cfg.DEVICE
        encoder, pool_layers, pool_dims = load_encoder_arch(self.enc_arch, self.n_pool_layers)
        encoder = encoder.to(device).eval()

        decoders = [load_decoder_arch(self.dec_arch, pool_dim) for pool_dim in pool_dims]
        decoders = [decoder.to(device) for decoder in decoders]

        params = list(decoders[0].parameters())
        for l in range(1, self.n_pool_layers):
            params += list(decoders[l].parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.LEARNING_RATE)
        N = 32
        print(f"Started training")
        for epoch_ix in tqdm(range(cfg.N_EPOCHS)):
            self.train_meta_epoch(epoch_ix, train_dataloader, encoder, decoders, optimizer, pool_layers, N, device)

        test_scores, test_gts, test_names = self.test_meta_epoch(test_dataloader, encoder, decoders, pool_layers, N, device, cfg.SAVE_SEGMENTATION, os.path.join(curr_run_path, "segmentation"))
        train_scores, train_gts, train_names = self.test_meta_epoch(train_dataloader, encoder, decoders, pool_layers, N, device, False, None)

        return_dict = {
            "tr_gts": np.array(train_gts), "tr_scores": train_scores, "tr_names": np.array(train_names),
            "ts_gts": np.array(test_gts), "ts_scores": test_scores, "ts_names": np.array(test_names),
        }
        return return_dict

    def test_meta_epoch(self, loader, encoder, decoders, pool_layers, N, device, save_segmentation, segmentation_save_dir):
        P = condition_vec
        decoders = [decoder.eval() for decoder in decoders]
        height = list()
        width = list()
        image_list = list()
        gt_label_list = list()
        gt_mask_list = list()
        test_dist = [list() for layer in pool_layers]
        test_loss = 0.0
        test_count = 0
        names = []
        with torch.no_grad():
            for i, sample_dict in enumerate(loader):
                image, label = sample_dict["image"], sample_dict["label"]
                names.extend(sample_dict["name"])
                gt_label_list.extend(t2np(label))
                # gt_mask_list.extend(t2np(mask))
                # data
                image = image.to(device)  # single scale
                _ = encoder(image)  # BxCxHxW
                # test decoder
                e_list = list()
                for l, layer in enumerate(pool_layers):
                    if 'vit' in self.enc_arch:
                        e = activation[layer].transpose(1, 2)[..., 1:]
                        e_hw = int(np.sqrt(e.size(2)))
                        e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                    else:
                        e = activation[layer]  # BxCxHxW
                    #
                    B, C, H, W = e.size()
                    S = H * W
                    E = B * S
                    #
                    if i == 0:  # get stats
                        height.append(H)
                        width.append(W)
                    #
                    p = positionalencoding2d(P, H, W).to(device).unsqueeze(0).repeat(B, 1, 1, 1)
                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    #
                    # m = F.interpolate(mask, size=(H, W), mode='nearest')
                    # m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                    #
                    decoder = decoders[l]
                    FIB = E // N + int(E % N > 0)  # number of fiber batches
                    for f in range(FIB):
                        if f < (FIB - 1):
                            idx = torch.arange(f * N, (f + 1) * N)
                        else:
                            idx = torch.arange(f * N, E)
                        #
                        c_p = c_r[idx]  # NxP
                        e_p = e_r[idx]  # NxC
                        # m_p = m_r[idx] > 0.5  # Nx1
                        #
                        if 'cflow' in self.dec_arch:
                            z, log_jac_det = decoder(e_p, [c_p, ])
                        else:
                            z, log_jac_det = decoder(e_p)
                        #
                        decoder_log_prob = get_logp(C, z, log_jac_det)
                        log_prob = decoder_log_prob / C  # likelihood per dim
                        loss = -log_theta(log_prob)
                        test_loss += t2np(loss.sum())
                        test_count += len(loss)
                        test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
        #
        test_map = [list() for p in pool_layers]
        for l, p in enumerate(pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm)  # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[l], width[l])
            # upsample
            test_map[l] = F.interpolate(test_mask.unsqueeze(1), size=self.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
        # score aggregation
        score_map = np.zeros_like(test_map[0])
        for l, p in enumerate(pool_layers):
            score_map += test_map[l]
        score_mask = score_map
        # invert probs to anomaly scores
        super_mask = score_mask.max() - score_mask
        # calculate detection AUROC
        score_label = np.max(super_mask, axis=(1, 2))
        gt_label = np.asarray(gt_label_list, dtype=np.bool)
        # det_roc_auc = roc_auc_score(gt_label, score_label)
        if save_segmentation:
            for img_b, seg, lbl, score, name in zip(loader.dataset, super_mask, gt_label, score_label, names):
                plot_sample(img_b["image"].detach().cpu().numpy().transpose((1, 2, 0)), None, seg, lbl, name, score, segmentation_save_dir, False, normalize_1=True)

        return score_label, gt_label, names

        # return height, width, image_list, test_dist, gt_label_list, gt_mask_list

    def train_meta_epoch(self, epoch_ix, dataloader, encoder, decoders, optimizer, pool_layers, N, device):
        decoders = [decoder.train() for decoder in decoders]
        self.adjust_learning_rate(optimizer, epoch_ix)
        I = len(dataloader)
        iterator = iter(dataloader)
        for sub_epoch in range(self.sub_epochs):
            for i in range(I):
                # warm-up learning rate
                self.warmup_learning_rate(epoch_ix, i + sub_epoch * I, I * self.sub_epochs, optimizer)
                # sample batch
                try:
                    sample_dict = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    sample_dict = next(iterator)
                # encoder prediction
                image = sample_dict["image"].to(device)  # single scale
                with torch.no_grad():
                    _ = encoder(image)
                # train decoder
                for l, layer in enumerate(pool_layers):
                    if 'vit' in self.enc_arch:
                        e = activation[layer].transpose(1, 2)[..., 1:]
                        e_hw = int(np.sqrt(e.size(2)))
                        e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                    else:
                        e = activation[layer].detach()  # BxCxHxW
                    #
                    B, C, H, W = e.size()
                    S = H * W
                    E = B * S
                    #
                    P = condition_vec
                    p = positionalencoding2d(P, H, W).to(device).unsqueeze(0).repeat(B, 1, 1, 1)
                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    perm = torch.randperm(E).to(device)  # BHW
                    decoder = decoders[l]
                    #
                    FIB = E // N  # number of fiber batches
                    assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                    for f in range(FIB):  # per-fiber processing
                        idx = torch.arange(f * N, (f + 1) * N)
                        c_p = c_r[perm[idx]]  # NxP
                        e_p = e_r[perm[idx]]  # NxC
                        if 'cflow' in self.dec_arch:
                            z, log_jac_det = decoder(e_p, [c_p, ])
                        else:
                            z, log_jac_det = decoder(e_p)
                        #
                        decoder_log_prob = get_logp(C, z, log_jac_det)
                        log_prob = decoder_log_prob / C  # likelihood per dim
                        loss = -log_theta(log_prob)
                        optimizer.zero_grad()
                        loss.mean().backward()
                        optimizer.step()

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.cfg.LEARNING_RATE
        if self.lr_cosine:
            eta_min = lr * (self.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / self.cfg.N_EPOCHS)) / 2
        else:
            steps = np.sum(epoch >= np.asarray(self.lr_decay_epochs))
            if steps > 0:
                lr = lr * (self.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_learning_rate(self, epoch, batch_id, total_batches, optimizer):
        if self.lr_warm and epoch < self.lr_warm_epochs:
            p = (batch_id + epoch * total_batches) / \
                (self.lr_warm_epochs * total_batches)
            lr = self.lr_warmup_from + p * (self.lr_warmup_to - self.lr_warmup_from)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        #
        for param_group in optimizer.param_groups:
            lrate = param_group['lr']
        return lrate
