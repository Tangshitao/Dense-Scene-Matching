import torch
import torch.utils.data as data
import os.path as osp

from dataset.dataset import *
from utils.transform import *


class Trainer(object):
    def __init__(self, cfg, model, logger):
        self.reproj_loss = cfg.TRAIN.reproj_loss
        self.reproj_loss_scale = cfg.TRAIN.reproj_loss_scale
        self.reproj_loss_start = cfg.TRAIN.reproj_loss_start

        transform = Compose(
            [
                Resize(
                    (
                        cfg.MODEL.TRANSFORM.train_resize_h,
                        cfg.MODEL.TRANSFORM.train_resize_w,
                    )
                ),
                Normalize(
                    scale=cfg.MODEL.TRANSFORM.scale,
                    mean=cfg.MODEL.TRANSFORM.mean,
                    std=cfg.MODEL.TRANSFORM.std,
                ),
            ]
        )

        if cfg.TRAIN.DATASET.type == "7scene":
            self.dataset = VideoDataset7scene(cfg.TRAIN.DATASET, transform)
        elif cfg.TRAIN.DATASET.type == "cambridge":
            self.dataset = VideoDatasetCambridge(cfg.TRAIN.DATASET, transform)
        elif cfg.TRAIN.DATASET.type == "scannet":
            self.dataset = VideoDatasetScannet(cfg.TRAIN.DATASET, transform)

        self.model = model
        self.logger = logger

        self.data_loader = data.DataLoader(
            self.dataset,
            cfg.TRAIN.batch_size,
            num_workers=cfg.TRAIN.workers,
            shuffle=True,
            pin_memory=True,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.TRAIN.base_lr, weight_decay=1e-4
        )
        self.base_lr = cfg.TRAIN.base_lr
        self.lr_steps1 = cfg.TRAIN.lr_steps1
        self.lr_steps2 = cfg.TRAIN.lr_steps2

        self.niter = 0
        self.data_loader_iter = iter(self.data_loader)
      
    def adjust_learning_rate(self):
        if self.niter in self.lr_steps1:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5

        if self.niter in self.lr_steps2:
            if self.niter == self.lr_steps2[0]:
                self.reproj_loss_start = True
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.base_lr * 0.5
            elif self.niter in self.lr_steps2:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.5

    def train_iters(self, iter_num):
        self.model.train()
        for i in range(iter_num):
            try:
                q, r = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                q, r = next(self.data_loader_iter)

            q_img = q["img"].cuda()
            q_Tcw = q["Tcw"].cuda()
            q_K = q["K"].cuda()
            q_depth = q["depth"].cuda()

            s_img = r["img"].cuda()
            s_Tcw = r["Tcw"].cuda()
            s_K = r["K"].cuda()
            s_depth = r["depth"].cuda()
          
            losses, metrics, pred_coords, gt_coords, _, _ = self.model(
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
            loss = losses["loss_cls"] + losses["loss_coords1"] + losses["loss_coords2"]

            losses["loss_reproj_q"] = losses["loss_reproj_q"] * self.reproj_loss_scale
            losses["loss_reproj_r"] = losses["loss_reproj_r"] * self.reproj_loss_scale

            self.adjust_learning_rate()
            if self.reproj_loss:
                loss = (
                    loss
                    + losses["loss_reproj_q"]
                    + losses["loss_reproj_r"]
                    + losses["loss_cls"]
                )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(
                "Iter: {}, loss: {:.3f}, loss_coords1: {:.3f}, loss_coords2: {:.3f}, loss_reproj1: {:.3f}, loss_reproj2: {:.3f}, lr: {}".format(
                    self.niter,
                    loss,
                    losses["loss_coords1"],
                    losses["loss_coords2"],
                    losses["loss_reproj_q"],
                    losses["loss_reproj_r"],
                    self.optimizer.param_groups[0]["lr"],
                )
            )
            self.write_metrics(metrics)
            losses["Avg_Loss"] = loss
            self.write_metrics(losses)
            self.niter += 1

    def write_metrics(self, metrics):
        if self.logger is not None:
            self.logger.add_metrics("train", self.niter, metrics)
