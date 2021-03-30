
//#include "nms_coords.h"
#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <stdio.h>
#include <torch/extension.h>
template <typename scalar_t>
__global__ void nms_kernel(
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords_2d_grid, //coords_2d_grid: N, H, W, M, 2
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> idxs, 
  int topk,
  int num_grid,
  int neighborhood_grid_width){
    

    int x1 = blockIdx.x;
    int y1 = blockIdx.y;
    int item = blockIdx.z;
   
    int idx_cnt=0;
    for(int i=0;i<coords_2d_grid.size(3)&&idx_cnt<topk;i++){

        scalar_t x2=coords_2d_grid[item][y1][x1][i][1];
        scalar_t y2=coords_2d_grid[item][y1][x1][i][0];
        int x2_round=round(x2);
        int y2_round=round(y2);
        bool flag=true;
        
        for(int j=0;j<i;j++){
          
          if(int(round(coords_2d_grid[item][y1][x1][j][1]))==x2_round \
            &&int(round(coords_2d_grid[item][y1][x1][j][0]))==y2_round \
            &&(coords_2d_grid[item][y1][x1][j][1]!=x2 \
            ||coords_2d_grid[item][y1][x1][j][0]!=y2)){ //ignore padding
            flag=false;
          }
          
        }
        if(flag){
          idxs[item][idx_cnt][y1][x1]=i;
          idx_cnt++;
        }
    }
  }

std::vector<torch::Tensor> nms_on_gpu(torch::Tensor coords_2d_grid, //coords_2d_grid: N, H, W, M, 2
    int topk,
    int max_displacement){
        int neighborhood_grid_width=2*max_displacement+1;
        int num_grid=neighborhood_grid_width*neighborhood_grid_width;
        const auto N = coords_2d_grid.size(0);
        const auto H = coords_2d_grid.size(1);
        const auto W = coords_2d_grid.size(2);
        const auto M = coords_2d_grid.size(3);
        
        auto idxs = torch::zeros({N, topk, H, W},torch::device(torch::kCUDA))-1;
        
        int shared_memory_per_block = 0;

        dim3 totalBlocksCorr(W, H, N);
        dim3 threadsPerBlock(1);

        AT_DISPATCH_FLOATING_TYPES(idxs.type(), "nms_kernel", ([&] {
            nms_kernel<scalar_t><<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(float)>>>(
                coords_2d_grid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                idxs.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                topk,
                num_grid,
                neighborhood_grid_width);
          }));
        return {idxs};
    }