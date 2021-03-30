import argparse
import yaml
import os
import numpy as np
import torch.utils.data as data
import torch
import tqdm

import sys

sys.path.append("libs")

from utils.base import *
from engine.launcher import *
from utils.transform import *
from model.arch.DSMNet import dsm_net
from model.geometry import (
    projection,
    back_projection,
    batched_scale_K,
    x_2d_coords_torch,
)
from utils.geometry import x_2d_coords
from dataset.dataset import VideoDataset7scene, VideoDatasetCambridge
from utils.geometry import rel_distance, rel_rot_angle, compute_pose_lm_pnp, scale_K


def pose_error(Tcw_gt, K, pred_coords, ori_h):
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
        Tcw_pred, inliner = compute_pose_lm_pnp(
            pred_coords[i], pnp_x_2d, scaled_K, 0.45, hypo=256
        )

        d_error = rel_distance(Tcw_pred, Tcw_gt[i])
        r_error = rel_rot_angle(Tcw_pred, Tcw_gt[i])
        d_errors.append(d_error)
        r_errors.append(r_error)
    return d_errors, r_errors, inliner, Tcw_pred


prev_q_img = None
prev_q_Tcw = None
prev_q_depth = None
prev_q_K = None
d_error_tempo_list = []
r_error_tempo_list = []


tot = 0
correct = 0
correct_tempo = 0
prev_idx = 0
skip_num = 1


def rel_dis_rot(Tcw1, Tcw2):
    Tcw1 = Tcw1[0, 0].cpu().numpy()
    Tcw2 = Tcw2[0, 0].cpu().numpy()
    d_err = rel_distance(Tcw1, Tcw2)
    r_err = rel_rot_angle(Tcw1, Tcw2)
    return d_err, r_err


def to_tensor(numpy_dict):
    tensor_dict = {}
    for k, v in numpy_dict.items():
        v = torch.as_tensor(v).unsqueeze(0).cuda()
        tensor_dict[k] = v
    return tensor_dict


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/cambridge.yaml")

args = parser.parse_args()
cfg = AttrDict(yaml.load(open(args.config)))

transform = Compose(
    [
        Resize((cfg.MODEL.TRANSFORM.test_resize_h, cfg.MODEL.TRANSFORM.test_resize_w)),
        Normalize(
            scale=cfg.MODEL.TRANSFORM.scale,
            mean=cfg.MODEL.TRANSFORM.mean,
            std=cfg.MODEL.TRANSFORM.std,
        ),
    ]
)
if cfg.TEST.DATASET.type == "cambridge":
    ds = VideoDatasetCambridge(cfg.TEST.DATASET, transform)
    skip_num = 1
elif cfg.TEST.DATASET.type == "7scene":
    ds = VideoDataset7scene(cfg.TEST.DATASET, transform)
    skip_num = 1
else:
    raise NotImplementedError

torch.manual_seed(0)
data_loader = data.DataLoader(ds, 1, num_workers=0, shuffle=False, pin_memory=True)
model = dsm_net(cfg.MODEL).cuda().eval()

for i in range(len(ds)):
    print(i)
    tot += 1
    q, r = ds[i]
    q = to_tensor(q)
    r = to_tensor(r)

    q_img = q["img"].cuda()
    q_Tcw = q["Tcw"].cuda()
    q_K = q["K"].cuda()
    q_depth = q["depth"].cuda()

    s_img = r["img"].cuda()
    s_Tcw = r["Tcw"].cuda()
    s_K = r["K"].cuda()
    s_depth = r["depth"].cuda()
    with torch.no_grad():
        losses, metrics, pred_coords, gt_coords, gt_mask, scores = model(
            q_img, q_depth, q_Tcw, q_K, s_img, s_depth, s_Tcw, s_K, s_Tcw[:, 0, :, :]
        )
    H, W = pred_coords.shape[-2:]

    d_errors, r_errors, inliner, Tcw_pred = pose_error(
        q_Tcw[:, 0, :, :], q_K[:, 0, :, :], pred_coords, q_img.size(3)
    )

    d_errors_tempo = d_errors
    r_errors_tempo = r_errors

    if prev_q_img is not None and i - prev_idx <= skip_num:
        q_img_tempo = torch.cat([q_img, prev_q_img], dim=1)
        q_Tcw_tempo = torch.cat([q_Tcw, prev_q_Tcw], dim=1)
        q_K_tempo = torch.cat([q_K, prev_q_K], dim=1)
        q_depth_tempo = torch.cat([q_depth, prev_q_depth], dim=1)
        with torch.no_grad():
            losses, metrics, pred_coords, gt_coords, gt_mask, scores = model(
                q_img_tempo,
                q_depth_tempo,
                q_Tcw_tempo,
                q_K_tempo,
                s_img,
                s_depth,
                s_Tcw,
                s_K,
                s_Tcw[:, 0, :, :],
                scores,
            )
        d_errors_tempo, r_errors_tempo, inliner_tempo, tempo_Tcw_pred = pose_error(
            q_Tcw[:, 0, :, :], q_K[:, 0, :, :], pred_coords, q_img.size(3)
        )

        if inliner.sum() > inliner_tempo.sum():
            d_errors_tempo = d_errors
            r_errors_tempo = r_errors
        else:
            Tcw_pred = tempo_Tcw_pred
            inliner = inliner_tempo

    r_errors_tempo = [min(r_errors_tempo[0], 360 - r_errors_tempo[0])]

    d_error_tempo_list.append(d_errors_tempo)
    r_error_tempo_list.append(r_errors_tempo)

    if i % skip_num == 0 and inliner.sum() / (H * W) > 0.01:
        prev_q_img = q_img
        prev_q_Tcw = torch.as_tensor(Tcw_pred.reshape(1, 1, 3, 4)).float().cuda()
        prev_q_depth = q_depth
        prev_q_K = q_K
        prev_idx = i
   
print(cfg.TEST.DATASET.seq_list_path)
print(
    "Median translation error: {}, Median rotation error: {}".format(
        np.median(d_error_tempo_list), np.median(r_error_tempo_list)
    )
)
