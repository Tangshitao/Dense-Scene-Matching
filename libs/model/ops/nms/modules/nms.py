import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from ..functions.nms import nms_op
from ....geometry import projection
import numpy as np

def nms_test(x_2d, topk, max_displacement):
    device=x_2d.device
    x_2d=x_2d.cpu().numpy()
    N,H,W,M,_=x_2d.shape
    idxs=np.zeros((N,topk,H,W))-1
    #import pdb
    #pdb.set_trace()
    for i in range(H):
        for j in range(W):
            x_2d_set=set([])
            cur_iter=0
            for m in range(M):
                x=x_2d[0,i,j,m,1]
                y=x_2d[0,i,j,m,0]
                if (x,y) not in x_2d_set:
                    idxs[0,cur_iter,i,j]=m
                    cur_iter+=1
                x_2d_set.add((x,y))
                if cur_iter>=topk:
                    break
    return torch.as_tensor(idxs,device=device)
                


class NMS_coords(Module):
    def __init__(self, topk, max_displacement):
        super(NMS_coords, self).__init__()
        self.topk=topk
        self.max_displacement=max_displacement

    def forward(self, coords_grid, anchor_P):
        N,M,_,H,W=coords_grid.shape
        coords_grid=coords_grid.permute(0,2,3,4,1).reshape(N,3,H*W,M).contiguous()
        x_2d,_=projection(coords_grid, anchor_P)
        
        x_2d=x_2d.permute(0,2,3,1).reshape(N,H,W,M,2).contiguous()
        
        idxs=nms_op(x_2d, self.topk, self.max_displacement)
       
        return idxs.long()
        

