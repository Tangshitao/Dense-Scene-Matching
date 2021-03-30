import torch.utils.data as data
import torch
import numpy as np
import os

from dataset.dataset import *
from utils.transform import *
from utils.geometry import *
from utils.logger import *


class Tester(object):
    def __init__(self, cfg, model, logger):
        transform = Compose(
            [
                Resize(
                    (
                        cfg.MODEL.TRANSFORM.test_resize_h,
                        cfg.MODEL.TRANSFORM.test_resize_w,
                    )
                ),
                Normalize(
                    scale=cfg.MODEL.TRANSFORM.scale,
                    mean=cfg.MODEL.TRANSFORM.mean,
                    std=cfg.MODEL.TRANSFORM.std,
                ),
            ]
        )
        if cfg.TEST.DATASET.type == "7scene":
            self.dataset = VideoDataset7scene(cfg.TEST.DATASET, transform)
        elif cfg.TEST.DATASET.type == "cambridge":
            self.dataset = VideoDatasetCambridge(cfg.TEST.DATASET, transform)
        elif cfg.TEST.DATASET.type == "scannet":
            self.dataset = VideoDatasetScannet(cfg.TEST.DATASET, transform)

        self.model = model
        self.logger = logger

        self.data_loader = data.DataLoader(
            self.dataset,
            cfg.TEST.batch_size,
            num_workers=cfg.TEST.workers,
            shuffle=False,
            pin_memory=True,
        )
        self.counter = Counter()
        self.niter = 0

    def pose_error(self, Tcw_gt, K, pred_coords, ori_h):
        pred_coords = pred_coords.permute(0, 2, 3, 1).cpu().detach().numpy()

        K = K.cpu().detach().numpy()
        scale = pred_coords.shape[1] / ori_h
        Tcw_gt = Tcw_gt.cpu().detach().numpy()

        N = K.shape[0]

        d_errors = []
        r_errors = []

        for i in range(N):
            scaled_K = scale_K(K[i], scale)

            H, W = pred_coords[i].shape[:2]
            pnp_x_2d = x_2d_coords(H, W)

            Tcw_pred, inliners = compute_pose_lm_pnp(
                pred_coords[i], pnp_x_2d, scaled_K, 1, hypo=256
            )

            d_error = rel_distance(Tcw_pred, Tcw_gt[i])
            r_error = rel_rot_angle(Tcw_pred, Tcw_gt[i])
            d_errors.append(d_error)
            r_errors.append(r_error)

        return d_errors, r_errors

    def run(self, max_iter=-1, eval_pose=False, write_metrics=False, results_path=None):
        self.model.eval()
        counter = Counter()
        iter_num = 0
        self.niter += 1
        print("run test, nter: {}".format(self.niter))
        for q, r in self.data_loader:
            if max_iter > 0 and iter_num >= max_iter:
                break
            q_img = q["img"].cuda()
            q_Tcw = q["Tcw"].cuda()
            q_K = q["K"].cuda()
            q_depth = q["depth"].cuda()

            s_img = r["img"].cuda()
            s_Tcw = r["Tcw"].cuda()
            s_K = r["K"].cuda()
            s_depth = r["depth"].cuda()

            with torch.no_grad():
                losses, metrics, pred_coords, gt_coords, gt_mask, scores = self.model(
                    q_img,
                    q_depth,
                    q_Tcw,
                    q_K,
                    s_img,
                    s_depth,
                    s_Tcw,
                    s_K,
                    s_Tcw[:, 0, :, :],
                )

            for k, v in metrics.items():
                counter.add_value(k, v)

            if eval_pose:
                d_errors, r_errors = self.pose_error(
                    q_Tcw[:, 0, :, :], q_K[:, 0, :, :], pred_coords, q_img.size(3)
                )
                print(d_errors)
                for d_error, r_error in zip(d_errors, r_errors):
                    if d_error < 0.05 and r_error < 5:
                        counter.add_value("rel_dist_acc", 1)
                    else:
                        counter.add_value("rel_dist_acc", 0)

                    counter.add_value("trans_pose_error", d_error, window_len=0)
                    counter.add_value("rot_pose_error", r_error, window_len=0)
            iter_num += 1

            print("test iter: {}".format(iter_num))

        if write_metrics:
            self.write_metrics(counter)

        return counter

    def write_metrics(self, metrics):
        if self.logger is not None:
            self.logger.add_metrics("val", self.niter, metrics)
