import torch
import torch.nn.functional as F


def flattenNL(tensor):
    N, L = tensor.shape[:2]
    tensor = tensor.reshape(N * L, *tensor.shape[2:])
    return tensor


def flattenNL_list(tensors):
    return [flattenNL(t) for t in tensors]


def expandNL(tensor, N, L):
    return tensor.reshape(N, L, *tensor.shape[1:])


def expandNL_list(tensors, N, L):
    return [expandNL(t, N, L) for t in tensors]


def get_euc_dis_error(coords, target_coords, mask):
    n, c, h, w = target_coords.shape
    n, c, h1, w1 = coords.shape
    if h1 != h or w1 != w:
        coords = F.interpolate(coords, size=(h, w), mode="bilinear", align_corners=True)
        if c == 2:
            coords = coords * (h / h1)
    dis_error = ((coords - target_coords) ** 2).sum(dim=1).sqrt()
    dis_error = (dis_error * mask).sum() / (mask.sum())
    return dis_error.detach()


def get_euc_dis_acc(coords, target_coords, mask):
    n, c, h, w = target_coords.shape
    n, c, h1, w1 = coords.shape
    if h1 != h or w1 != w:
        coords = F.interpolate(coords, size=(h, w), mode="bilinear", align_corners=True)
        if c == 2:
            coords = coords * (h / h1)
    dis_error = ((coords - target_coords) ** 2).sum(dim=1).sqrt()
    mask_inside_window = (dis_error < 0.05) * mask
    acc = mask_inside_window.sum() / mask.sum()
    return acc.detach()


def get_reproj_acc(reproj_error, reproj_target, mask, thresh):
    n, c, h, w = reproj_error.shape
    n, c, h1, w1 = reproj_target.shape
    if h != h1:
        reproj_error = F.interpolate(
            reproj_error, scale_factor=h1 // h, mode="bilinear", align_corners=True
        )
    mask_inside_window = (
        ((reproj_error - reproj_target) ** 2).sum(dim=1).sqrt() < thresh
    ).float() * mask

    acc = mask_inside_window.sum() / mask.sum()

    return acc.detach()


def gen_dummy_input():
    q_img = torch.randn(1, 1, 3, 192, 256)
    q_depth = torch.randn(1, 1, 192, 256)
    q_Tcw = torch.randn(1, 1, 3, 4)
    q_K = torch.randn(1, 1, 3, 3)

    s_img = torch.randn(1, 3, 3, 192, 256)
    s_depth = torch.randn(1, 3, 192, 256)
    s_Tcw = torch.randn(1, 3, 3, 4)
    s_K = torch.randn(1, 3, 3, 3)

    return [q_img, q_depth, q_Tcw, q_K, s_img, s_depth, s_Tcw, s_K]


def gen_dummy_head_input(cfg):
    feat = torch.randn(1, 256, 6, 8)
    corr = torch.randn(1, cfg.topk, 6, 8)
    s_coords_grid = torch.randn(1, cfg.topk * 3, 6, 8)
    p_coords = torch.randn(1, 3, 6, 8)

    return [feat, corr, s_coords_grid, p_coords]
