import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from ..functions.corr import correlation_op, correlation_proj_op
from ....geometry import projection
import torch.nn.functional as F


class Correlation(Module):

    def __init__(self, pad_size=None, kernel_size=None, max_displacement=None,
                 stride1=None, stride2=None, corr_multiply=None):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def reset_params(self):
        return

    def forward(self, input1, input2):
        return correlation_op(input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

    def __repr__(self):
        return self.__class__.__name__

class CorrelationProj(Module):
    
    def __init__(self, max_displacement=None, stride=1):
        super(CorrelationProj, self).__init__()
        self.max_displacement = max_displacement
        self.stride = stride

    def reset_params(self):
        return

    def forward(self, input1, input2, query_coords, scene_coords, scene_P):
        input1=input1.permute(0,2,3,1).contiguous()
        input2=input2.permute(0,1,3,4,2).contiguous()
        return correlation_proj_op(input1, input2, query_coords, scene_coords, scene_P, self.max_displacement, self.stride)

    def __repr__(self):
        return self.__class__.__name__


class CorrelationPytorch(Module):
    def __init__(self, max_displacement=None, stride=1):
        super().__init__()
        self.max_displacement = max_displacement
        self.stride = stride

    def query_scene_corr(self, query_feat, scene_feat, query_proj_2d):
        """
        query_feat: N,C,H,W
        scene_feat: N,L,C,H,W
        query_proj_2d: N*L,H,W,2
        """
        _,_,H,W=query_feat.shape
        N, L, C, rH, rW = scene_feat.shape
        scene_feat = F.grid_sample(scene_feat.reshape(
            N*L, -1, rH, rW), query_proj_2d,mode='nearest', align_corners=True)
        
        scene_feat = scene_feat.reshape(
            N, L, C, H*W).permute(0, 3, 1, 2).reshape(-1, L, C)  # N*H*W,L,C
        query_feat = query_feat.permute(0, 2, 3, 1).reshape(-1, C, 1)

        corr = torch.bmm(scene_feat, query_feat)
        corr = corr.reshape(N, H, W, L).permute(0, 3, 1, 2)
        return corr

    def normalize_coordinates(self, coords, h, w):
        h = h-1
        w = w-1
        coords = coords.clone().data
        coords[:, :, :, 0] -= w / 2
        coords[:, :, :, 1] -= h / 2
        coords[:, :, :, 0] /= w / 2
        coords[:, :, :, 1] /= h / 2
        return coords

    def forward(self, input1, input2, query_coords, scene_coords, scene_P):
        """
        input1: N,C1,H,W
        input2: N,L,C1,H,W
        query_coords: N,3,H,W
        scene_coords: N,L,3,H,W
        scene_P: N,L,3,4
        """
        
        N, L, C1, rH, rW = input2.shape
        _,_,H,W=input1.shape
        
        with torch.no_grad():
            
            query_proj_2d, valid_mask = projection(
                query_coords, scene_P, clip=False)
            
            query_proj_2d = query_proj_2d.permute(0,1,3,4,2).reshape(-1,H,W,2)
            
            query_proj_2d = torch.round(query_proj_2d)
        
        corr1_tensor = []
        coords_tensor = []
        mask_tensor=[]
        for i in range(-self.max_displacement, self.max_displacement+1):
            for j in range(-self.max_displacement, self.max_displacement+1):
                query_proj_2d_shift = query_proj_2d.clone()
                query_proj_2d_shift[:, :, :, 0] += j*self.stride
                query_proj_2d_shift[:, :, :, 1] += i*self.stride
                
                query_proj_2d_shift = self.normalize_coordinates(
                    query_proj_2d_shift, rH, rW)
                
                corr1 = self.query_scene_corr(
                    input1, input2, query_proj_2d_shift)
                
                scene_coords_sample = F.grid_sample(
                    scene_coords.reshape(-1,3,rH,rW), query_proj_2d_shift,mode='nearest', align_corners=True).reshape(N,-1,3,H,W)

                mask=((query_proj_2d_shift[:,:,:,0]>=-1)*(query_proj_2d_shift[:,:,:,0]<=1)*(query_proj_2d_shift[:,:,:,1]>=-1)*(query_proj_2d_shift[:,:,:,1]<=1)).float()
                corr1_tensor.append(corr1)
                coords_tensor.append(scene_coords_sample)
                mask_tensor.append(mask.reshape(N,L,H,W))
        
        corr1_tensor = torch.cat(corr1_tensor, dim=1)
        mask_tensor =torch.cat(mask_tensor,dim=1)
        coords_tensor = torch.cat(coords_tensor, dim=1)

        return corr1_tensor, coords_tensor, mask_tensor

def test_corr_cuda():
    import time
    torch.set_printoptions(precision=10)
    corr1=CorrelationPytorch(max_displacement=4).cuda()
    corr2=CorrelationProj(max_displacement=4).cuda()

    tensors=torch.load('/home/shitaot/slam-projects/video_camera_localization/corr_tensors.pth')
    i=0
    l=0
    query_feat=tensors['q_feat'].cuda()[:,:,:,:].contiguous().detach().clone()
    scene_feat=tensors['s_feat'].cuda().contiguous().detach().clone()
    
    rough_coords=tensors['r_coords'].cuda()[:,:,:,:].contiguous()
    scene_coords_norm=tensors['s_coords_norm'].cuda().contiguous()
    print(scene_coords_norm.shape)
    s_P=tensors['s_P'].cuda().contiguous()
    s_P=torch.zeros_like(s_P,device=s_P.device)
    s_P[:,:,0,0]=1
    s_P[:,:,1,1]=1
    s_P[:,:,2,2]=1
    #conv= nn.Conv2d(128, 128,1).cuda()
    #scene_feat=conv(scene_feat.reshape(-1,128,12,16)).reshape(scene_feat.shape)
    #query_feat=conv(query_feat)

    #print(scene_feat.shape, query_feat.shape, rough_coords.shape, scene_coords_norm.shape, s_P.shape)
    t1=time.time()
    query_feat.requires_grad=True
    scene_feat.requires_grad=True
    c1, coords1, mask1=corr1(query_feat, scene_feat, rough_coords, scene_coords_norm, s_P)
    t2=time.time()
    query_feat_copy=query_feat.detach().clone()
    scene_feat_copy=scene_feat.detach().clone()
    query_feat_copy.requires_grad=True
    scene_feat_copy.requires_grad=True
    c2,coords2, mask2=corr2(query_feat_copy, scene_feat_copy, rough_coords, scene_coords_norm, s_P)
    #print(c1.shape,c2.shape,coords1.shape,coords2.shape)
    # print((c1[0,:,3,3]-c2[0,:,3,3]).abs().mean())
    # print(c2[0,:,3,3])
    
    
    def register_grad(v):
        def register(grad):
            v.grad_nonleaf=grad
        v.register_hook(register)
    register_grad(c1)
    register_grad(c2)
    error=(c1-c2)/(c1+1e-22)
    mask=coords2!=0
    error2=(coords1-coords2)*mask
    error3=mask1-mask2
    _corr=query_feat[22,:,10,11]*scene_feat[22,0,:,2,4]
    _corr=_corr.reshape(8,32).sum(dim=0).cpu().detach().numpy()
    s=0
    s1=0
    for i in range(32):
        s+=_corr[i]
        s1+=_corr[31-i]
        print(_corr[i], s, s1)
    print(error.mean(),error.max(), error2.sum()/(mask.sum()), error3.sum())
    
    loss1=(c1**2).sum()
    t1=time.time()
    loss1.backward()
    t2=time.time()
    # import pdb
    # pdb.set_trace()
    
    loss2=(c2**2).sum()
    t3=time.time()
    loss2.backward()
    t4=time.time()
    m1=(query_feat.grad==0).float()
    m2=(query_feat_copy.grad==0).float()
    assert((m1-m2).sum()==0)
    diff=(scene_feat.grad-scene_feat_copy.grad).abs()/(scene_feat.grad+1e-20)
    diff2=(query_feat.grad-query_feat_copy.grad).abs()/(query_feat.grad+1e-20)
    #assert((query_feat.grad-query_feat_copy.grad).abs().max()<1e-5)
    print(diff.abs().max(), diff2.abs().max())
    print(scene_feat.grad[ 27,   0, 151,   1,   4], scene_feat_copy.grad[ 27,   0, 151,   1,   4])
    import pdb
    pdb.set_trace()
    

if __name__=="__main__":
    test_corr_cuda()
    import time
    corr1=CorrelationPytorchMultiView(max_displacement=4).cuda()

    tensors=torch.load('/home/shitaot/my-pwc-net/PyTorch/data/correlation_test.pth')
    i=0
    l=0
    query_feat=tensors['q_feat'].cuda()#[:,:,:,:].contiguous().detach().clone()
    scene_feat=tensors['s_feat'].cuda()#[:,:,:,:,:].contiguous().detach().clone()
    rough_coords=tensors['r_coords'].cuda()#[:,:,:,:].contiguous()
    scene_coords_norm=tensors['s_coords_norm'].cuda()#[:,:,:,:,:].contiguous()
    s_P=tensors['s_P'].cuda()#[:,:,:,:].contiguous()

    corr1(query_feat, scene_feat, rough_coords, scene_coords_norm, s_P, s_P[:,0,:,:])

