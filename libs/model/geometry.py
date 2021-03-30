import numpy as np
import torch


def x_2d_coords_torch(n, h, w, dim=1, device=None):
    if dim == 1:
        x_2d = torch.zeros((n, 2, h, w), device=device)
        for y in range(0, h):
            x_2d[:, 1, y, :] = y
        for x in range(0, w):
            x_2d[:, 0, :, x] = x
    elif dim == -1:
        x_2d = np.zeros((n, h, w, 2), device=device)
        for y in range(0, h):
            x_2d[:, y, :, 1] = y
        for x in range(0, w):
            x_2d[:, :, x, 0] = x
    return x_2d


def batched_scale_K(K, rescale_factor):
    K = K * rescale_factor
    K[:, 2, 2] = 1.0
    return K


def batched_transpose(R, t, X):
    """
    Pytorch batch version of computing transform of the 3D points
    :param R: rotation matrix in dimension of (N, 3, 3)
    :param t: translation vector (N, 3)
    :param X: points with 3D position, a 2D array with dimension of (N, 3, M)
    :return: transformed 3D points
    """
    assert R.size(1) == 3
    assert R.size(2) == 3
    assert t.size(1) == 3
    N = R.size(0)
    X = torch.bmm(R, X)
    X = X + t.view(N, 3, 1)
    return X


def batched_pi_inv(K, x, d):
    """
    Projecting the pixel in 2D image plane and the depth to the 3D point in camera coordinate.
    :param x: 2d pixel position, a 2D array with dimension of (N, num_points, 2)
    :param d: depth at that pixel, a array with dimension of (N, num_points, 1)
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :return: 3D point in camera coordinate (N, num_points, 3)
    """
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]

    X_x = d * (x[:, 0:1, :] - cx) / fx
    X_y = d * (x[:, 1:2, :] - cy) / fy
    X_z = d
    X = torch.cat([X_x, X_y, X_z], dim=1)
    return X


def batched_inv_pose(R, t):
    N = R.size(0)
    Rwc = torch.transpose(R, 1, 2)
    tw = -torch.bmm(Rwc, t.view(N, 3, 1))
    return Rwc, tw


def back_projection(depth, Tcw, K):
    n, h, w = depth.size()

    x_2d = x_2d_coords_torch(n, h, w, device=depth.device).view(n, 2, -1)

    depth = depth.view(n, 1, h * w)
    X_3d = batched_pi_inv(K, x_2d, depth)
    Rwc, twc = batched_inv_pose(R=Tcw[:, :3, :3], t=Tcw[:, :3, 3])
    X_3d = batched_transpose(Rwc, twc, X_3d)
    X_3d = X_3d.reshape(n, 3, h, w)

    return X_3d


def projection(coords, P, ori_H=None, ori_W=None, scale_factor=1.0, clip=False):
    if len(P.shape) == 3:
        return projection3D(coords, P, ori_H, ori_W, scale_factor=1.0, clip=clip)
    elif len(P.shape) == 4:
        return projection4D(coords, P, ori_H, ori_W, scale_factor=1.0, clip=clip)
    else:
        raise NotImplementedError


def projection3D(coords, P, ori_H=None, ori_W=None, scale_factor=1.0, clip=True):
    """
    coords: N,3,H,W
    P: N,L,3,4
    """

    B, _, H, W = coords.size()
    if ori_H is None:
        ori_H = H
    if ori_W is None:
        ori_W = W
    coords = coords.view(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=coords.device)

    coords_homo = torch.cat([coords, ones], dim=1)

    P = P.reshape(-1, 3, 4)
    coords_2d = torch.bmm(P, coords_homo)

    if clip:
        z = torch.clamp(coords_2d[:, 2:3, :], min=0.1)
    else:
        z = coords_2d[:, 2:3, :] + 1e-5
    valid_mask = z.reshape(B, H, W) > 1e-5
    coords_2d = coords_2d[:, :2, :] / (z)

    coords_2d = coords_2d / scale_factor

    coords_2d = coords_2d.view(B, 2, H, W)

    valid_mask = (
        valid_mask
        * (
            (coords_2d[:, 0, :, :] >= 0)
            * (coords_2d[:, 1, :, :] >= 0)
            * (coords_2d[:, 0, :, :] <= ori_W - 1)
            * (coords_2d[:, 1, :, :] <= ori_H - 1)
        ).float()
    )

    return coords_2d, valid_mask


def projection4D(coords, P, ori_H=None, ori_W=None, scale_factor=1.0, clip=True):
    """
    coords: N,3,H,W
    P: N,L,3,4
    """

    N, L, _, _ = P.shape
    _, _, H, W = coords.shape
    if ori_H is None:
        ori_H = H
    if ori_W is None:
        ori_W = W

    dev_id = coords.device
    coords = coords.reshape(N, 3, -1)
    coords_homo = torch.cat([coords, torch.ones(N, 1, H * W).to(dev_id)], dim=1)

    P = P.reshape(N, L * 3, 4)
    coords_2d = torch.bmm(P, coords_homo)
    coords_2d = coords_2d.reshape(N, L, 3, -1)
    if clip:
        z = torch.clamp(coords_2d[:, :, -1:, :], min=0.1)
    else:
        z = coords_2d[:, :, -1:, :] + 1e-5
    valid_mask = z.reshape(N, L, H, W) > 1e-5
    coords_2d = (coords_2d / (z))[:, :, :2, :].reshape(N, L, 2, H, W) / scale_factor

    valid_mask = (
        valid_mask
        * (
            (coords_2d[:, :, 0, :, :] >= 0)
            * (coords_2d[:, :, 1, :, :] >= 0)
            * (coords_2d[:, :, 0, :, :] <= ori_W - 1)
            * (coords_2d[:, :, 1, :, :] <= ori_H - 1)
        ).float()
    )

    return coords_2d, valid_mask
