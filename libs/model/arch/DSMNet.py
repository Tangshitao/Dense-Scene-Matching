import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry import batched_scale_K, projection, back_projection, x_2d_coords_torch
from ..ops.correlation.modules.corr import (
    CorrelationProj,
    Correlation,
    CorrelationPytorch,
)
from ..ops.nms.modules.nms import NMS_coords
from ..head import GeneralHead, confHead, ContextNorm
from ..basic import (
    flattenNL,
    expandNL,
    get_euc_dis_error,
    get_reproj_acc,
    get_euc_dis_acc,
)
from ..backbone.fpn import (
    build_resnet_fpn_backbone,
    build_resnet18_fpn_backbone,
    build_vgg_fpn_backbone,
)
from ..backbone.resnet import build_resnet18_backbone
import numpy as np


class APLoss(nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq - 1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(
            a * min + np.arange(nq, 0, -1)
        )  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(
            np.arange(2 - nq, 2, 1) - a * min
        )  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret="1-mAP"):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, : self.nq], q[:, self.nq :]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(
            dim=-1
        )  # number of correct samples = c+ N x Q

        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= 1e-16 + rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == "1-mAP":
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == "AP":
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {"loss_ap": float(loss)}


class DSMNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, cfg):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        self.md = cfg.max_displacement
        self.topk = cfg.topk
        self.max_pyramid = cfg.max_pyramid

        super(DSMNet, self).__init__()

        self.freeze_backbone = cfg.freeze_backbone

        self.corr_proj = CorrelationProj(max_displacement=self.md)

        self.nms = NMS_coords(self.topk, self.md)

        self.extractor = build_resnet_fpn_backbone(cfg.BACKBONE)

        self.head5 = GeneralHead(cfg, 256, cfg.topk * 4, 0)

        self.head4 = GeneralHead(cfg, 256, cfg.topk * 4, 3)

        self.head3 = GeneralHead(cfg, 256, cfg.topk * 4, 3)

        self.head2 = GeneralHead(cfg, 256, cfg.topk * 4, 3)

        self.head1 = GeneralHead(cfg, 256, cfg.topk * 4, 3)

        self.leakyRELU = nn.LeakyReLU(0.1)

        self.conf_head = confHead(dim=4)
        self.conf_head2 = confHead(dim=4)
        self.conf_head3 = confHead(dim=4)
        self.conf_head4 = confHead(dim=4)

        self.output_tensor = {}
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

        self.ap_loss_m = APLoss()

    def load_extractor_pretrain(self, path, bottom_up_pretrain=False, cfg=None):
        if bottom_up_pretrain:
            self.extractor.load_bottom_up_pretrain(path, cfg)
        else:
            self.extractor.load_pretrain(path)

    def avg_tensor(self, tensor, mask, kernel_size=2):
        mask = mask.unsqueeze(1)
        tensor = F.avg_pool2d(tensor * mask, kernel_size=kernel_size)
        mask = F.avg_pool2d(mask.float(), kernel_size=kernel_size)
        tensor = tensor / (mask + 1e-5)
        mask = mask > 0
        tensor = tensor * mask
        return tensor

    def avg_depth(self, depth, scale):
        N, H, W = depth.shape
        mask = depth > 1e-5

        depth = self.avg_tensor(
            depth.unsqueeze(1), mask, kernel_size=int(1 / scale)
        ).squeeze(1)

        return depth

    def normalize_2d_coords(self, x_2d, dim=3):
        if dim == 3:
            N, H, W, _ = x_2d.shape
            h = H - 1
            w = W - 1
            x_2d[:, :, :, 0] = x_2d[:, :, :, 0] - w / 2
            x_2d[:, :, :, 1] = x_2d[:, :, :, 1] - h / 2
            x_2d[:, :, :, 0] = x_2d[:, :, :, 0] / (w / 2)
            x_2d[:, :, :, 1] = x_2d[:, :, :, 1] / (h / 2)
        else:
            N, _, H, W = x_2d.shape
            h = H - 1
            w = W - 1
            x_2d[:, 0, :, :] = x_2d[:, 0, :, :] - w / 2
            x_2d[:, 1, :, :] = x_2d[:, 1, :, :] - h / 2
            x_2d[:, 0, :, :] = x_2d[:, 0, :, :] / (w / 2)
            x_2d[:, 1, :, :] = x_2d[:, 1, :, :] / (h / 2)
        return x_2d

    def fill_holes(self, feat, coords, mask, feat_up, coords_up, fill_feat=True):
        coords_up = F.interpolate(coords_up, scale_factor=2, mode="nearest")

        mask = mask.unsqueeze(1)
        if fill_feat:
            feat_up = F.interpolate(feat_up, scale_factor=2, mode="nearest")
            feat = feat * mask + (1 - mask) * feat_up

        coords = coords * mask + (1 - mask) * coords_up
        return feat, coords

    def fuse_and_fill_holes_pyramid(
        self, feat_pyramid, coords_pyramid, mask_pyramid, s_P_pyramid
    ):
        feat_pyramid = [
            self.fuse_feat(f, c, s_P)[0]
            for f, c, s_P in zip(feat_pyramid, coords_pyramid, s_P_pyramid)
        ]

        N, L = feat_pyramid[0].shape[:2]
        f_up, c_up = flattenNL(feat_pyramid[0]), flattenNL(coords_pyramid[0])
        new_feat_pyramid = [feat_pyramid[0]]
        new_coords_pyramid = [coords_pyramid[0]]
        new_mask_pyramid = [mask_pyramid[0]]

        for f, c, m in zip(feat_pyramid[1:], coords_pyramid[1:], mask_pyramid[1:]):
            f = flattenNL(f)
            c = flattenNL(c)
            m = flattenNL(m)
            f, c = self.fill_holes(f, c, m, f_up, c_up, fill_feat=False)
            new_feat_pyramid.append(expandNL(f, N, L))
            new_coords_pyramid.append(expandNL(c, N, L))
            new_mask_pyramid.append(expandNL(m, N, L))

            f_up = f
            c_up = c
        return new_feat_pyramid, new_coords_pyramid, new_mask_pyramid

    def gen_coords_list(self, depth, Tcw, K, feat_h, feat_w, pyramid_num=4):
        """
        depth: N, H, W
        Tcw: N, 3, 4
        K: N, 3, 3
        """
        N, H, W = depth.shape

        scale = feat_h / H
        depth = self.avg_depth(depth, scale)

        K = batched_scale_K(K, scale)
        coords = back_projection(depth, Tcw, K)
        mask = (depth > 1e-5).float()
        coords_list = [coords]
        mask_list = [mask]
        for i in range(pyramid_num):
            K = batched_scale_K(K, 0.5)
            depth = self.avg_tensor(depth.unsqueeze(1), mask, 2).squeeze(1)
            mask = F.avg_pool2d(mask, kernel_size=2) > 0
            mask = mask.float()
            coords = back_projection(depth, Tcw, K)
            coords_list.insert(0, coords)
            mask_list.insert(0, mask)

        return list(reversed(coords_list)), list(reversed(mask_list))

    def build_coords_pyramid(
        self, q_depth, q_Tcw, q_K, s_depth, s_Tcw, s_K, feat_h, feat_w
    ):
        """
        q_depth: N, T, H, W
        q_Tcw: N, T, 3, 4
        q_K: N, T, 3, 3
        s_depth: N, L, H, W
        s_Tcw: N, L, 3, 4
        s_K: N, L, 3, 3
        """

        N, L, H, W = s_depth.shape

        depth = torch.cat([q_depth, s_depth], dim=1)
        Tcw = torch.cat([q_Tcw, s_Tcw], dim=1)
        K = torch.cat([q_K, s_K], dim=1)

        coords_list, mask_list = self.gen_coords_list(
            depth.view(-1, H, W),
            Tcw.view(-1, 3, 4),
            K.view(-1, 3, 3),
            feat_h,
            feat_w,
            self.max_pyramid,
        )

        q_coords_pyramid = []
        s_coords_pyramid = []
        q_mask_pyramid = []
        s_mask_pyramid = []

        for c, m in zip(
            coords_list[-self.max_pyramid - 1 :], mask_list[-self.max_pyramid - 1 :]
        ):
            _, _, h, w = c.shape
            c = c.view(N, -1, 3, h, w)
            m = m.view(N, -1, h, w)

            q_coords_pyramid.insert(0, c[:, :-L, :, :, :].contiguous())
            s_coords_pyramid.insert(0, c[:, -L:, :, :, :].contiguous())
            q_mask_pyramid.insert(0, m[:, :-L, :, :].contiguous())
            s_mask_pyramid.insert(0, m[:, -L:, :, :].contiguous())

        return q_coords_pyramid, s_coords_pyramid, q_mask_pyramid, s_mask_pyramid

    def build_feat_pyramid(self, q_img, s_img):
        N, L, C, H, W = s_img.shape
        T = q_img.shape[1]
        img = torch.cat([q_img, s_img], dim=1)
        img = img.view(-1, *img.shape[-3:])
        feat = self.extractor(img)

        q_feat_pyramid = []
        s_feat_pyramid = []
        for i in range(6 - self.max_pyramid, 7):
            f = feat["p{}".format(i)]
            f = f.reshape(N, T + L, *f.shape[-3:])
            q_feat_pyramid.insert(0, f[:, :T, :, :, :])
            s_feat_pyramid.insert(0, f[:, T:, :, :, :])

        return q_feat_pyramid, s_feat_pyramid

    def grid_tensor(self, tensor):
        B, C, H, W = tensor.size()

        with torch.no_grad():
            ones = torch.ones(B, 1, H, W, device=tensor.device)
            tensor_list = []
            for i in range(C):
                tensor_list.append(
                    self.corr(ones, tensor[:, i : i + 1, :, :].contiguous()).unsqueeze(
                        2
                    )
                )
            output = torch.cat(tensor_list, dim=2)
            return output

    def sort_corr(self, corr, coords, mask, r_P):
        N, C, H, W = corr.shape

        corr, idx = torch.sort(corr, dim=1, descending=True)
        mask = torch.gather(mask, dim=1, index=idx)

        idx_3d = idx.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        coords = coords.reshape(N, -1, 3, H, W)
        coords = torch.gather(coords, dim=1, index=idx_3d).contiguous()

        idx_nms = self.nms(coords, r_P)
        idx_mask = (idx_nms != -1).long()
        idx_nms = idx_mask * idx_nms

        corr = torch.gather(corr, dim=1, index=idx_nms) * idx_mask
        mask = torch.gather(mask, dim=1, index=idx_nms) * idx_mask
        idx_nms = idx_nms.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        coords = torch.gather(coords, dim=1, index=idx_nms) * idx_mask.unsqueeze(2)
        return corr, coords, mask

    def retrieve_topk(
        self, q_feat_norm, s_feat_norm, s_coords, s_P, r_P, p_coords=None
    ):
        N, L, C, H, W = s_feat_norm.shape
        if p_coords is None:
            q_feat_norm = q_feat_norm.reshape(N, C, -1)
            s_feat_norm = s_feat_norm.permute(0, 1, 3, 4, 2).reshape(N, -1, C)
            corr = torch.bmm(s_feat_norm, q_feat_norm).reshape(N, -1, H, W)
            s_coords_grid = (
                s_coords.permute(0, 1, 3, 4, 2)
                .reshape(N, -1, 3, 1, 1)
                .repeat(1, 1, 1, H, W)
            )
            mask = torch.ones(corr.shape, device=corr.device)
        else:
            corr, s_coords_grid, mask = self.corr_proj(
                q_feat_norm, s_feat_norm, p_coords, s_coords, s_P
            )

        return self.sort_corr(corr, s_coords_grid, mask, r_P)

    def fuse_feat(self, feat, s_coords, s_P, self_mask=True):

        N, L, C, H, W = feat.shape
        scene_feat_list = []
        mask_list = []
        for l in range(s_coords.shape[1]):
            x_2d, m = projection(s_coords[:, l, :, :, :], s_P)

            if self_mask:
                m[:, l, :, :] = 1
            x_2d = x_2d.reshape(-1, 2, H, W)
            x_2d = x_2d.permute(0, 2, 3, 1)

            x_2d = self.normalize_2d_coords(x_2d, dim=3)

            scene_align_feat = F.grid_sample(
                feat.reshape(N * L, -1, H, W), x_2d, mode="bilinear", align_corners=True
            )
            scene_align_feat = scene_align_feat.reshape(N, L, -1, H, W)

            scene_align_feat = scene_align_feat * m.unsqueeze(2)

            scene_feat_avg = scene_align_feat.sum(dim=1) / (
                m.sum(dim=1, keepdim=True) + 1e-7
            )

            scene_feat_list.append(scene_feat_avg.unsqueeze(1))
            mask_list.append(m)
        scene_feat = torch.cat(scene_feat_list, dim=1)
        mask = torch.cat(mask_list, dim=1)
        return scene_feat, mask

    def fuse_tempo_feat(self, feat, s_coords, s_P, self_mask=True):
        N, L, C, H, W = feat.shape
        scene_feat_list = []
        mask_list = []
        for l in range(s_coords.shape[1]):
            x_2d, m = projection(s_coords[:, l, :, :, :], s_P)
            # m=(x_2d[:,:,0,:,:]>-0.5)*(x_2d[:,:,0,:,:]<W-0.5)*(x_2d[:,:,1,:,:]>-0.5)*(x_2d[:,:,1,:,:]<H-0.5)
            if self_mask:
                m[:, l, :, :] = 1
            x_2d = x_2d.reshape(-1, 2, H, W)
            x_2d = x_2d.permute(0, 2, 3, 1)

            x_2d = self.normalize_2d_coords(x_2d, dim=3)

            scene_align_feat = F.grid_sample(
                feat.reshape(N * L, -1, H, W), x_2d, mode="bilinear", align_corners=True
            )

            scene_align_feat = scene_align_feat.reshape(N, L, -1, H, W)

            scene_align_feat = scene_align_feat * m.unsqueeze(2)

            scene_feat_avg = scene_align_feat.sum(dim=1) / (
                m.sum(dim=1, keepdim=True) + 1e-7
            )

            scene_feat_list.append(scene_feat_avg.unsqueeze(1))
            mask_list.append(m)
        scene_feat = torch.cat(scene_feat_list, dim=1)
        mask = torch.cat(mask_list, dim=1)
        return scene_feat, mask

    def predict_conf(self, x, K, conf_head):

        N, C, H, W = x.shape
        Tcw_eye = torch.zeros(N, 3, 4, device=K.device)
        Tcw_eye[:, :3, :3] = torch.eye(3, device=x.device)
        x_2d_pred, _ = projection(x.reshape(-1, C, H, W), Tcw_eye)

        x = x.reshape(N, -1, H, W)
        x_2d = x_2d_coords_torch(N, H, W, device=x.device)
        x_2d[:, 0, :, :] -= W / 2
        x_2d[:, 1, :, :] -= H / 2

        x_2d /= K[:, 0, 0].reshape(N, 1, 1, 1)

        x = torch.cat([x, x_2d], dim=1)

        return conf_head(x)

    def predict_coords(
        self,
        q_feat_combined,
        s_feat,
        s_coords,
        q_P,
        s_P,
        r_P,
        head,
        p_coords=None,
        q_K=None,
        conf_head=None,
        scores=None,
    ):
        N, L, C, H, W = s_coords.shape

        T = q_feat_combined.shape[1]
        q_feat = q_feat_combined[:, 0, :, :, :].contiguous()
        q_feat_norm = F.normalize(q_feat, p=2, dim=1)
        s_feat_norm = F.normalize(s_feat, p=2, dim=2)

        if T > 1:
            q_temporal_feat = q_feat_combined[:, 1:, :, :]
            q_temporal_P = q_P[:, 1:, :, :]

            q_feat_fuse, mask = self.fuse_tempo_feat(
                q_temporal_feat, s_coords, q_temporal_P, False
            )

            mask = mask.unsqueeze(2)
            q_feat_fuse_norm = F.normalize(q_feat_fuse, p=2, dim=2)

            scale = torch.clamp(scores.mean(), max=0.6) + 0.4

            s_feat_norm = (
                s_feat_norm * scale + q_feat_fuse_norm * (1 - scale)
            ) * mask + s_feat_norm * (1 - mask)

        corr, s_coords_grid, mask = self.retrieve_topk(
            q_feat_norm, s_feat_norm, s_coords, s_P, r_P, p_coords
        )
        s_coords_grid_ori = s_coords_grid

        mean = s_coords.view(N, L, C, -1).mean(dim=-1).mean(dim=1)
        std = (
            (s_coords.view(N, L, C, -1) - mean.view(N, 1, 3, 1))
            .view(N, -1)
            .std(dim=-1, keepdim=True)
        )

        s_coords_grid = (s_coords_grid - mean.view(N, 1, 3, 1, 1)) / (
            std.view(N, 1, 1, 1, 1) + 1e-5
        )

        if p_coords is not None:
            p_coords = (p_coords - mean.view(N, 3, 1, 1)) / (
                std.reshape(N, 1, 1, 1) + 1e-5
            )

            s_coords_grid = (s_coords_grid - p_coords.unsqueeze(1)) * mask.unsqueeze(2)
            s_coords_grid = s_coords_grid.reshape(N, -1, H, W)

        s_coords_grid = s_coords_grid.view(N, -1, H, W)

        pred_coords1, pred_coords2, scores = head(q_feat, corr, s_coords_grid, p_coords)

        def unnormalize_coords(coords, mean, std):
            coords = coords * std.reshape(N, 1, 1, 1)
            coords = coords + mean.reshape(N, 3, 1, 1)

            return coords

        pred_coords1 = unnormalize_coords(pred_coords1, mean, std)
        pred_coords2 = unnormalize_coords(pred_coords2, mean, std)

        if q_K is not None:
            scores = self.predict_conf(pred_coords1.detach(), q_K, conf_head)

        return pred_coords1, pred_coords2, scores, s_coords_grid_ori, std

    def predict_coords_pyramid(
        self,
        q_feat_pyramid,
        s_feat_pyramid,
        s_coords_pyramid,
        q_P_pyramid,
        s_P_pyramid,
        r_P_pyramid,
        q_K_pyramid=None,
        conf=None,
    ):
        head_list = [self.head5, self.head4, self.head3, self.head2]
        conf_head_list = [
            self.conf_head4,
            self.conf_head3,
            self.conf_head2,
            self.conf_head,
        ]

        p_coords = None
        pred_coords1_pyramid = []
        pred_coords2_pyramid = []
        std_pyramid = []
        s_coords_grid_pyramid = []
        score_pyramid = []
        for i in range(self.max_pyramid):
            if p_coords is not None:
                p_coords = F.interpolate(p_coords, scale_factor=2, mode="bilinear")

            (
                pred_coords1,
                pred_coords2,
                scores,
                s_coords_grid,
                std,
            ) = self.predict_coords(
                q_feat_pyramid[i],
                s_feat_pyramid[i],
                s_coords_pyramid[i],
                q_P_pyramid[i],
                s_P_pyramid[i],
                r_P_pyramid[i],
                head_list[i],
                p_coords,
                q_K_pyramid[i],
                conf_head_list[i],
                conf[-1] if conf is not None else None,
            )
            p_coords = pred_coords1.detach()
            pred_coords1_pyramid.append(pred_coords1)
            pred_coords2_pyramid.append(pred_coords2)
            std_pyramid.append(std)
            s_coords_grid_pyramid.append(s_coords_grid)
            score_pyramid.append(scores)
        return (
            pred_coords1_pyramid,
            pred_coords2_pyramid,
            score_pyramid,
            s_coords_grid_pyramid,
            std_pyramid,
        )

    def gen_projection_matrix_pyramid(
        self, q_K, q_Tcw, s_K, s_Tcw, q_feat_pyramid, ori_h, ori_w
    ):
        N, L = s_K.shape[:2]
        r_Tcw = torch.zeros(N, 3, 4, device=q_K.device)
        r_Tcw[:, :3, :3] = torch.eye(3, device=q_K.device)

        q_P_pyramid = []
        r_P_pyramid = []
        s_P_pyramid = []

        q_K_pyramid = []

        for q_feat in q_feat_pyramid:
            N, T, C, H, W = q_feat.shape
            L = s_K.shape[1]
            scale = H / ori_h
            _q_K = expandNL(batched_scale_K(flattenNL(q_K), scale), N, q_K.size(1))
            _s_K = batched_scale_K(flattenNL(s_K), scale)
            q_P = expandNL(torch.bmm(flattenNL(_q_K), flattenNL(q_Tcw)), N, T)
            r_P = torch.bmm(_q_K[:, 0, :, :], r_Tcw)
            s_P = expandNL(torch.bmm(_s_K.view(-1, 3, 3), s_Tcw.view(-1, 3, 4)), N, L)
            q_P_pyramid.append(q_P)
            r_P_pyramid.append(r_P)
            s_P_pyramid.append(s_P)
            q_K_pyramid.append(_q_K[:, 0, :, :])
        return q_P_pyramid, r_P_pyramid, s_P_pyramid, q_K_pyramid

    def projection(self, coords_pyramid, P_pyramid):
        coords_2d_pyramid = []
        for i in range(self.max_pyramid):
            coords_2d, m = projection(coords_pyramid[i], P_pyramid[i], clip=True)
            coords_2d_pyramid.append(coords_2d)
        return coords_2d_pyramid

    def relative_Tcw(self, anchor_Tcw, Tcw):
        R = anchor_Tcw[:, :3, :3]
        T = anchor_Tcw[:, :3, 3:]
        R_inv = R.permute(0, 2, 1)
        T_inv = torch.bmm(-R_inv, T)
        L = Tcw.shape[1]
        R_inv = R_inv.unsqueeze(1).repeat(1, L, 1, 1)
        T_inv = T_inv.unsqueeze(1).repeat(1, L, 1, 1)
        Tcw_R = Tcw[:, :, :3, :3]
        Tcw_T = Tcw[:, :, :3, 3:]
        Tcw_T = Tcw_T.reshape(-1, 3, 1) + torch.bmm(
            Tcw_R.reshape(-1, 3, 3), T_inv.reshape(-1, 3, 1)
        )
        Tcw_R = torch.bmm(Tcw_R.reshape(-1, 3, 3), R_inv.reshape(-1, 3, 3))
        new_Tcw = torch.cat([Tcw_R, Tcw_T], dim=2).reshape(-1, L, 3, 4)

        return new_Tcw

    def rot_coords(self, anchor_Tcw, coords):
        N, _, H, W = coords.shape
        coords = coords.reshape(N, 3, -1)
        R = anchor_Tcw[:, :3, :3]
        T = anchor_Tcw[:, :3, 3:]
        R_inv = R.permute(0, 2, 1)
        T_inv = torch.bmm(-R_inv, T)
        coords = torch.bmm(R_inv, coords) + T_inv
        coords = coords.reshape(N, 3, H, W)
        return coords

    def forward(
        self,
        q_img,
        q_depth,
        q_Tcw,
        q_K,
        s_img,
        s_depth,
        s_Tcw,
        s_K,
        r_Tcw,
        scores=None,
        inliner_ratios=None,
    ):
        N, L, H, W = s_depth.shape

        if self.freeze_backbone:
            with torch.no_grad():
                q_feat_pyramid, s_feat_pyramid = self.build_feat_pyramid(q_img, s_img)
        else:
            q_feat_pyramid, s_feat_pyramid = self.build_feat_pyramid(q_img, s_img)

        q_Tcw = self.relative_Tcw(r_Tcw.clone(), q_Tcw)
        s_Tcw = self.relative_Tcw(r_Tcw.clone(), s_Tcw)

        _, _, _, feat_h, feat_w = q_feat_pyramid[-1].shape
        (
            q_coords_pyramid,
            s_coords_pyramid,
            q_mask_pyramid,
            s_mask_pyramid,
        ) = self.build_coords_pyramid(
            q_depth, q_Tcw, q_K, s_depth, s_Tcw, s_K, feat_h, feat_w
        )

        (
            q_P_pyramid,
            r_P_pyramid,
            s_P_pyramid,
            q_K_pyramid,
        ) = self.gen_projection_matrix_pyramid(
            q_K, q_Tcw, s_K, s_Tcw, q_feat_pyramid, H, W
        )

        q_feat_pyramid = q_feat_pyramid[1:]
        q_coords_pyramid = q_coords_pyramid[1:]
        q_mask_pyramid = q_mask_pyramid[1:]
        q_P_pyramid = q_P_pyramid[1:]
        r_P_pyramid = r_P_pyramid[1:]
        s_P_pyramid = s_P_pyramid[1:]
        s_feat_pyramid = s_feat_pyramid[1:]
        s_coords_pyramid = s_coords_pyramid[1:]
        s_mask_pyramid = s_mask_pyramid[1:]

        (
            s_feat_pyramid,
            s_coords_pyramid,
            s_mask_pyramid,
        ) = self.fuse_and_fill_holes_pyramid(
            s_feat_pyramid, s_coords_pyramid, s_mask_pyramid, s_P_pyramid
        )

        (
            pred_coords1_pyramid,
            pred_coords2_pyramid,
            score_pyramid,
            s_coords_grid_pyramid,
            std_pyramid,
        ) = self.predict_coords_pyramid(
            q_feat_pyramid,
            s_feat_pyramid,
            s_coords_pyramid,
            q_P_pyramid,
            s_P_pyramid,
            r_P_pyramid,
            q_K_pyramid,
            scores,
        )

        s_coords = back_projection(
            s_depth.reshape(-1, H, W), s_Tcw.reshape(-1, 3, 4), s_K.reshape(-1, 3, 3)
        ).reshape(N, -1, 3, H, W)
        q_coords = back_projection(
            q_depth[:, 0, :, :], q_Tcw[:, 0, :, :], q_K[:, 0, :, :]
        )

        gt_coords_pyramid = [q[:, 0, :, :, :].contiguous() for q in q_coords_pyramid]
        gt_mask_pyramid = [m[:, 0, :, :].contiguous() for m in q_mask_pyramid]

        q_P_pyramid = [q_P[:, 0, :, :] for q_P in q_P_pyramid]

        losses, metrics = self.losses_and_metrics(
            pred_coords1_pyramid,
            pred_coords2_pyramid,
            score_pyramid,
            s_coords_grid_pyramid,
            gt_coords_pyramid,
            gt_mask_pyramid,
            q_P_pyramid,
            r_P_pyramid,
            std_pyramid,
        )

        pred_coords = self.rot_coords(r_Tcw, pred_coords1_pyramid[-1])
        gt_coords = self.rot_coords(r_Tcw, gt_coords_pyramid[-1])

        return (
            losses,
            metrics,
            pred_coords,
            gt_coords,
            gt_mask_pyramid[-1],
            torch.sigmoid(score_pyramid[-1]),
        )

    def loss(
        self,
        preds,
        gts,
        masks,
        std_pyramid=None,
        loss_scales=1,
        do_normalize=False,
        l2_loss=False,
        clip_value=-1,
    ):
        if isinstance(loss_scales, int):
            loss_scales = [loss_scales] * len(preds)

        err = 0.0
        errors = []
        i = 0
        for pred, gt, mask, s in zip(preds, gts, masks, loss_scales):
            if len(mask.shape) != len(pred.shape):
                mask = mask.unsqueeze(1).repeat(1, gt.size(1), 1, 1)
            if do_normalize:
                pred = F.normalize(pred, p=2, dim=1)
                gt = F.normalize(gt, p=2, dim=1)
            if std_pyramid is not None:
                pred = pred / std_pyramid[i].reshape(-1, 1, 1, 1)
                gt = gt / std_pyramid[i].reshape(-1, 1, 1, 1)

            l1_err = (pred * mask - gt * mask).abs()
            if l2_loss:
                l1_err = l1_err ** 2
                l1_err = l1_err * mask
            if clip_value > 0:
                l1_err = torch.clamp(l1_err, max=clip_value)

            error = s * l1_err.sum() / torch.sum(mask)
            err += error
            errors.append(error)
        return err, errors

    def cls_loss(self, scores_pyramid, pred_coords1_pyramid, q_P_pyramid):
        loss_sum = 0
        tot = 0
        loss2_sum = 0
        for scores, pred_coords, q_p in zip(
            scores_pyramid, pred_coords1_pyramid, q_P_pyramid
        ):

            n, _, h, w = pred_coords.shape

            x_2d, _ = projection(pred_coords, q_p)
            x_2d = x_2d.reshape(n, 2, -1, h, w)

            x_2d_gt = x_2d_coords_torch(n, h, w, device=x_2d.device)

            reproj_err = (x_2d - x_2d_gt.unsqueeze(2)).abs().sum(dim=1)

            gt_label = reproj_err < 1
            scores = torch.sigmoid(scores)
            loss = self.ap_loss_m(scores.reshape(n, -1), gt_label.reshape(n, -1))

            scores_ranked, idx = torch.sort(
                scores.reshape(n, -1), dim=1, descending=True
            )
            gt_label_ranked = torch.gather(gt_label.reshape(n, -1), dim=1, index=idx)

            self.output_tensor["cls_gt_label"] = gt_label

            loss_sum += loss
            tot += 1

        return loss_sum / tot

    def losses_and_metrics(
        self,
        pred_coords1_pyramid,
        pred_coords2_pyramid,
        score_pyramid,
        s_coords_grid_pyramid,
        gt_coords_pyramid,
        gt_mask_pyramid,
        q_P_pyramid,
        r_P_pyramid,
        std_pyramid,
    ):
        pred_2d_coords_q_P_pyramid = self.projection(pred_coords1_pyramid, q_P_pyramid)
        pred_2d_coords_r_P_pyramid = self.projection(pred_coords1_pyramid, r_P_pyramid)

        gt_2d_coords_q_P_pyramid = self.projection(gt_coords_pyramid, q_P_pyramid)
        gt_2d_coords_r_P_pyramid = self.projection(gt_coords_pyramid, r_P_pyramid)

        loss_cls = self.cls_loss(score_pyramid, pred_coords1_pyramid, q_P_pyramid)

        loss_scales = [0.8, 0.9, 1.0, 1.1, 1.1]
        loss_coords1, c = self.loss(
            pred_coords1_pyramid,
            gt_coords_pyramid,
            gt_mask_pyramid,
            std_pyramid=std_pyramid,
            loss_scales=loss_scales,
        )
        loss_coords2, d = self.loss(
            pred_coords2_pyramid,
            gt_coords_pyramid,
            gt_mask_pyramid,
            std_pyramid=std_pyramid,
            loss_scales=loss_scales,
        )

        loss_scales_reproj = [1.0, 1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 8]

        loss_scales_reproj = [a * b for a, b in zip(loss_scales, loss_scales_reproj)]

        loss_reproj_q, a = self.loss(
            pred_2d_coords_q_P_pyramid,
            gt_2d_coords_q_P_pyramid,
            gt_mask_pyramid,
            clip_value=10,
            loss_scales=loss_scales,
        )
        loss_reproj_r, b = self.loss(
            pred_2d_coords_r_P_pyramid,
            gt_2d_coords_r_P_pyramid,
            gt_mask_pyramid,
            clip_value=10,
            loss_scales=loss_scales,
        )
        # print(loss_reproj_q, loss_reproj_r)
        losses = {
            "loss_coords1": loss_coords1,
            "loss_coords2": loss_coords2,
            "loss_reproj_q": loss_reproj_q,
            "loss_reproj_r": loss_reproj_r,
            "loss_cls": loss_cls,
        }

        metrics = {}

        reproj_threshs = [0.75 / 8, 0.75 / 4, 0.75 / 2, 0.75]
        for i in range(self.max_pyramid):
            euc_dis_error = get_euc_dis_error(
                pred_coords1_pyramid[i], gt_coords_pyramid[i], gt_mask_pyramid[i]
            )

            euc_dis_error2 = get_euc_dis_error(
                pred_coords2_pyramid[i], gt_coords_pyramid[i], gt_mask_pyramid[i]
            )

            euc_dis_acc = get_euc_dis_acc(
                pred_coords2_pyramid[i], gt_coords_pyramid[i], gt_mask_pyramid[i]
            )

            reproj_acc = get_reproj_acc(
                pred_2d_coords_q_P_pyramid[i],
                gt_2d_coords_q_P_pyramid[i],
                gt_mask_pyramid[i],
                reproj_threshs[i],
            )

            reproj_error = get_euc_dis_error(
                pred_2d_coords_q_P_pyramid[i],
                gt_2d_coords_q_P_pyramid[i],
                gt_mask_pyramid[i],
            )

            reproj_error_s = get_euc_dis_error(
                pred_2d_coords_r_P_pyramid[i],
                gt_2d_coords_r_P_pyramid[i],
                gt_mask_pyramid[i],
            )

            idx = 5 - i
            metrics["euc_dis_error{}".format(idx)] = euc_dis_error
            metrics["euc_dis_acc{}".format(idx)] = euc_dis_acc
            metrics["euc_dis_error{}_2".format(idx)] = euc_dis_error2
            metrics["reproj_acc{}".format(idx)] = reproj_acc
            metrics["reproj_error{}".format(idx)] = reproj_error
            metrics["reproj_error_s{}".format(idx)] = reproj_error_s
        return losses, metrics


def dsm_net(cfg):
    model = DSMNet(cfg)
    if "feat_pretrained_path" in cfg and cfg.feat_pretrained_path is not None:
        model.load_extractor_pretrain(
            cfg.feat_pretrained_path, cfg.bottom_up_pretrain, cfg.BACKBONE
        )
    if "model_path" in cfg and cfg.model_path is not None:
        print("Load model from {}".format(cfg.model_path))
        data = torch.load(cfg.model_path)
        if "state_dict" in data.keys():
            data = data["state_dict"]
      
        state_dict = {}
        for k, v in model.state_dict().items():
            l = len(v.size())
            flag = True
            if k in data:
                for i in range(l):
                    if v.size(i) != data[k].size(i):
                        flag = False
                if flag:
                    state_dict[k] = data[k]
                else:
                    state_dict[k] = model.state_dict()[k]
            else:
                state_dict[k] = model.state_dict()[k]

        model.load_state_dict(state_dict)

    return model
