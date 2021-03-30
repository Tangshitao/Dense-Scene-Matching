import nms
import torch.nn as nn

from torch.autograd import Function

class NMS(Function):
    @staticmethod
    def forward(ctx, coords_2d_grid, topk, max_displacement):
        # import pdb
        # pdb.set_trace()
        return nms.nms_forward(coords_2d_grid, topk, max_displacement)[0]

nms_op=NMS.apply