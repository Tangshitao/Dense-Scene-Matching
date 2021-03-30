import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv_bn(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    bias=True,
    require_grads=True,
    lr_mul=1,
    load_weights=True,
    forward_hook=False,
):
    cv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )

    bn = nn.BatchNorm2d(out_planes)
    bn.freeze = not require_grads
    for param in cv.parameters():
        param.requires_grad = require_grads
        param.lr_mul = lr_mul
        param.load_weights = load_weights
    for param in bn.parameters():
        param.requires_grad = require_grads
        param.lr_mul = lr_mul
        param.load_weights = load_weights

    return nn.Sequential(cv, bn, nn.ReLU())


class ContextNorm(nn.Module):
    def __init__(self, width):
        super(ContextNorm, self).__init__()

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x = (x - mean) / (var + 1e-5).sqrt()
        return x


class confHead(nn.Module):
    def __init__(self, dim=5, out=1):
        super(confHead, self).__init__()
        self.conv1 = conv1x1(5, 128)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.res1 = Bottleneck(128, 128, cn=ContextNorm(128))
        self.res2 = Bottleneck(128, 128, cn=ContextNorm(128))
        self.res3 = Bottleneck(128, 128, cn=ContextNorm(128))
        self.res4 = Bottleneck(128, 128, cn=ContextNorm(128))
        self.res5 = Bottleneck(128, 128, cn=ContextNorm(128))
        self.res6 = Bottleneck(128, 128, cn=ContextNorm(128))
        self.conf = conv1x1(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.conf(x)
        return x


class PredictionDenseHead(nn.Module):
    def __init__(self, dim, topk, freeze=False, lr_mul=1):
        super().__init__()
        dd = list(np.cumsum([128, 128, 96, 64, 32]))
        # self.neck = conv_bn(
        #      dim,      32, kernel_size=3, stride=1, require_grads=not freeze, lr_mul=lr_mul)

        od = dim
        self.conv0_c = conv_bn(
            od, 128, kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv1_c = conv_bn(
            od + dd[0], 128, kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv2_c = conv_bn(
            od + dd[1], 96, kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv3_c = conv_bn(
            od + dd[2], 64, kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv4_c = conv_bn(
            od + dd[3], 32, kernel_size=3, stride=1, require_grads=not freeze
        )
        self.predict_coords = conv3x3(od + dd[4], 3, bias=True)

    def forward(self, x):
        x = torch.cat((self.conv0_c(x), x), 1)
        x = torch.cat((self.conv1_c(x), x), 1)
        x = torch.cat((self.conv2_c(x), x), 1)
        x = torch.cat((self.conv3_c(x), x), 1)
        x = torch.cat((self.conv4_c(x), x), 1)
        x = self.predict_coords(x)
        return x


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        width=256,
        allconv1=False,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        cn=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.cn = cn
        if allconv1:
            self.conv2 = conv1x1(width, width)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.cn is not None:
            out = self.cn(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.cn is not None:
            out = self.cn(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.cn is not None:
            out = self.cn(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PredictionResHead(nn.Module):
    def __init__(self, dim, cfg, freeze=False, lr_mul=1):
        od = dim
        super().__init__()

        planes = cfg.res_channel_num
        width = planes * cfg.res_width_expand
        self.conv1 = conv_bn(
            od, planes, kernel_size=1, stride=1, require_grads=not freeze, padding=0
        )

        self.res1 = Bottleneck(planes, planes, width, allconv1=True)
        self.res2 = Bottleneck(planes, planes, width, allconv1=True)
        self.res3 = Bottleneck(planes, planes, width, allconv1=True)
        self.predict_coords = conv1x1(planes, 3, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        coords = self.predict_coords(x)
        return coords, x


class PredictionHead(nn.Module):
    def __init__(self, dim, feat_dim, topk, cfg, freeze=False, lr_mul=1):
        super().__init__()
        dd = list(np.cumsum(cfg.dense_dim))
        dense_dim = cfg.dense_dim

        self.neck2 = None
        print(feat_dim)
        self.neck = conv_bn(
            feat_dim,
            cfg.feat_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            require_grads=not freeze,
            lr_mul=lr_mul,
        )
        if cfg.feat_out_dim2 != -1:
            self.neck2 = conv_bn(
                cfg.feat_out_dim + dim,
                cfg.feat_out_dim2,
                kernel_size=1,
                stride=1,
                padding=0,
                require_grads=not freeze,
                lr_mul=lr_mul,
            )
            od = cfg.feat_out_dim2
        else:
            od = cfg.feat_out_dim + dim
        self.conv0_c = conv_bn(
            od, dense_dim[0], kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv1_c = conv_bn(
            od + dd[0], dense_dim[1], kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv2_c = conv_bn(
            od + dd[1], dense_dim[2], kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv3_c = conv_bn(
            od + dd[2], dense_dim[3], kernel_size=3, stride=1, require_grads=not freeze
        )
        self.conv4_c = conv_bn(
            od + dd[3], dense_dim[4], kernel_size=3, stride=1, require_grads=not freeze
        )
        self.predict_coords = conv3x3(od + dd[4], 3, bias=True)
        # self.predict_scores = conv3x3(
        #     od+dd[4], 1, bias=True)

    def forward(self, x1, x2):
        if self.neck is not None:
            x1 = self.neck(x1)
        x = torch.cat([x1, x2], dim=1)
        if self.neck2 is not None:
            x = self.neck2(x)

        x = torch.cat((self.conv0_c(x), x), 1)
        x = torch.cat((self.conv1_c(x), x), 1)
        x = torch.cat((self.conv2_c(x), x), 1)
        x = torch.cat((self.conv3_c(x), x), 1)
        x = torch.cat((self.conv4_c(x), x), 1)
        coords = self.predict_coords(x)
        # scores= self.predict_scores(x)
        return coords  # , scores


class GeneralHead(nn.Module):
    def __init__(self, cfg, feat_dim, dim, prev_coords=0):
        super().__init__()
        self.res_head = PredictionResHead(dim, cfg.HEAD)
        # self.res_head_score = PredictionResHead(dim, cfg.HEAD)
        self.dense_head = PredictionHead(
            cfg.HEAD.res_channel_num + prev_coords, feat_dim, cfg.topk, cfg.HEAD
        )

    def forward(self, feat, corr, s_coords_grid, p_coords):
        x = torch.cat([corr, s_coords_grid], dim=1)

        pred_coords2, x = self.res_head(x)

        if p_coords is not None:
            pred_coords2 = pred_coords2 + p_coords
            x = torch.cat([p_coords, x], dim=1)

        pred_coords1 = self.dense_head(feat, x)

        return pred_coords1, pred_coords2, None
