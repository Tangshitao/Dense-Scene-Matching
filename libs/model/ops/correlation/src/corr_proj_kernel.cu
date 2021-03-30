#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include "corr_proj.h"
#include <stdio.h>

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t

// == Dimension rearrangement Kernel

// == Correlation Kernel

template <typename scalar_t>
__global__ void CorrelateData(torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input1, //query t: N, H, W, C1
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input2, // scene : N, L, H, W, C1
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> query_coords,// query coords: N, 3, H, W
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> scene_coords, // scene coords: N, L, 3, H, W
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> scene_P, //projection matrix: N, L, 3, 4
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> corr1, // corr1 : N, M, H, W
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> return_coords, // coords : N, M, 3, H, W
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> mask, // corr1 : N, M, H, W
  const int neighborhood_grid_radius,
  const int neighborhood_grid_width,
  int stride){
  extern __shared__ char patch_data_char[];
  
  scalar_t *feat1_data = (scalar_t *)patch_data_char;

  int x1 = blockIdx.x;
  int y1 = blockIdx.y;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;
  

  for(int ch = ch_off; ch < input1.size(3); ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
    feat1_data[ch] = input1[item][y1][x1][ch];
  }
  __syncthreads();

  __shared__ scalar_t sum[WARPS_PER_BLOCK*THREADS_PER_WARP];
  scalar_t x_3d=query_coords[item][0][y1][x1];
  scalar_t y_3d=query_coords[item][1][y1][x1];
  scalar_t z_3d=query_coords[item][2][y1][x1];
  for(int l=0;l<input2.size(1);l++){
    
    scalar_t center_x3d=scene_P[item][l][0][0]*x_3d+scene_P[item][l][0][1]*y_3d+scene_P[item][l][0][2]*z_3d+scene_P[item][l][0][3];
    scalar_t center_y3d=scene_P[item][l][1][0]*x_3d+scene_P[item][l][1][1]*y_3d+scene_P[item][l][1][2]*z_3d+scene_P[item][l][1][3];
    scalar_t center_z3d=scene_P[item][l][2][0]*x_3d+scene_P[item][l][2][1]*y_3d+scene_P[item][l][2][2]*z_3d+scene_P[item][l][2][3];
    int center_x=round(center_x3d/(center_z3d+1e-5));
    int center_y=round(center_y3d/(center_z3d+1e-5));

    int min_x = center_x -neighborhood_grid_radius* stride;
    int max_x = center_x + neighborhood_grid_radius * stride;
    int min_y = center_y - neighborhood_grid_radius * stride;
    int max_y = center_y + neighborhood_grid_radius * stride;
    min_x = min_x>=0?min_x:min_x+((-min_x-1)/stride+1)*stride;
    max_x = max_x<input2.size(3)?max_x:(input2.size(3)-1);
    min_y = min_y>=0?min_y:min_y+((-min_y-1)/stride+1)*stride;
    max_y = max_y<input2.size(2)?max_y:(input2.size(2)-1);
    
    for(int y2 = min_y; y2 <= max_y; y2+=stride){
      for(int x2 = min_x; x2 <= max_x; x2+=stride){
        sum[ch_off]=0;
        for(int ch = ch_off; ch < input1.size(3); ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) {
          sum[ch_off]+=feat1_data[ch]*input2[item][l][y2][x2][ch];
          
        }

        if(ch_off==0){
          scalar_t total_sum = 0;
          for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
              total_sum += sum[idx];
          }
          int m = (((y2-center_y)/stride+neighborhood_grid_radius)*neighborhood_grid_width+(x2-center_x)/stride+neighborhood_grid_radius)*input2.size(1)+l;
          
          corr1[item][m][y1][x1]=total_sum;
          mask[item][m][y1][x1]=1;
          for(int i=0;i<3;i++){
            return_coords[item][m][i][y1][x1]=scene_coords[item][l][i][y2][x2];
          }
        }

      }
      
    }
  }
  // Aggregate  
}


std::vector<torch::Tensor> CorrelateData_ongpu(torch::Tensor input1, //query t: N, H, W, C1
  torch::Tensor input2, //scene : N, L, H, W, C1
  torch::Tensor query_coords, // scene coords: N, 3, H, W
  torch::Tensor scene_coords, // scene coords: N, L, 3, H, W
  torch::Tensor scene_P, //sceneTcw*K: N, L, 3, 4
  int max_displacement,
  int stride){

    const auto N = input1.size(0);
    const auto H = input1.size(1);
    const auto W = input1.size(2);
    const auto C1 = input1.size(3);
    const auto L = input2.size(1);
    const int neighborhood_grid_radius_ = max_displacement/stride;
    const int neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;


    auto corr1 = torch::zeros({N, L*neighborhood_grid_width_*neighborhood_grid_width_, H, W},torch::device(torch::kCUDA));
    
    auto return_coords = torch::zeros({N, L*neighborhood_grid_width_*neighborhood_grid_width_, 3, H, W},torch::device(torch::kCUDA));
    
    auto mask = torch::zeros({N, L*neighborhood_grid_width_*neighborhood_grid_width_, H, W},torch::device(torch::kCUDA));
    
    int shared_memory_per_block = C1;
    
    dim3 totalBlocksCorr(W, H, N);
    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input1.type(), "CorrelateData_ongpu", ([&] {
      CorrelateData<scalar_t><<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(scalar_t)>>>(
          input1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          input2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          query_coords.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          scene_coords.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          scene_P.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          corr1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          return_coords.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          mask.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          neighborhood_grid_radius_,
          neighborhood_grid_width_,
          stride);
    }));
  return {corr1, return_coords, mask};

}

template <typename scalar_t>
__global__ void CorrelateDataBackward(
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> corr_grad, //corr grad: N, M, H, W
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input1, //query t: N, H, W, C1
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input2, // scene : N, L, H, W, C1
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> query_coords,// query coords: N, 3, H, W
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> scene_P, //projection matrix: N, L, 3, 4
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input1_grad, //N, H, W, C1
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input2_grad, //N, L, H, W, C1
  const int neighborhood_grid_radius,
  const int neighborhood_grid_width,
  int stride,
  int item){
  extern __shared__ char patch_data_char[];
  
  

  int x1 = blockIdx.x;
  int y1 = blockIdx.y;
  int c = blockIdx.z;
  int ch_off = threadIdx.x;


  __shared__ scalar_t sum[THREADS_PER_WARP];
  sum[ch_off]=0;

  scalar_t x_3d=query_coords[item][0][y1][x1];
  scalar_t y_3d=query_coords[item][1][y1][x1];
  scalar_t z_3d=query_coords[item][2][y1][x1];
  for(int m=ch_off;m<corr_grad.size(1);m+=THREADS_PER_WARP){
    int l=m%input2.size(1);
    scalar_t center_x3d=scene_P[item][l][0][0]*x_3d+scene_P[item][l][0][1]*y_3d+scene_P[item][l][0][2]*z_3d+scene_P[item][l][0][3];
    scalar_t center_y3d=scene_P[item][l][1][0]*x_3d+scene_P[item][l][1][1]*y_3d+scene_P[item][l][1][2]*z_3d+scene_P[item][l][1][3];
    scalar_t center_z3d=scene_P[item][l][2][0]*x_3d+scene_P[item][l][2][1]*y_3d+scene_P[item][l][2][2]*z_3d+scene_P[item][l][2][3];
    int center_x=round(center_x3d/(center_z3d+1e-5));
    int center_y=round(center_y3d/(center_z3d+1e-5));

    int x2=(m/input2.size(1)%neighborhood_grid_width-neighborhood_grid_radius)*stride+center_x;
    int y2=(m/input2.size(1)/neighborhood_grid_width-neighborhood_grid_radius)*stride+center_y;
   
    if(y2>=0&&y2<input2.size(2)&&x2>=0&&x2<input2.size(3)){ 
      
      sum[ch_off] += input2[item][l][y2][x2][c]*corr_grad[item][m][y1][x1];
      atomicAdd(&input2_grad[item][l][y2][x2][c], corr_grad[item][m][y1][x1]*input1[item][y1][x1][c]);
    }
  }
  
  if(ch_off==0){
    scalar_t total_sum = 0;
    for(int idx = 0; idx < THREADS_PER_WARP; idx++) {
        total_sum += sum[idx];
    }
    input1_grad[item][y1][x1][c]=total_sum;
  }

  // Aggregate  
}

std::vector<torch::Tensor> CorrelateData_backward_ongpu(torch::Tensor grad_output1, //query t: N, H, W, C1
  torch::Tensor input1, //query : N,  H, W, C1
  torch::Tensor input2, // scene : N, L, H, W, C1
  torch::Tensor query_coords, // query coords: N, L, 3, H, W
  torch::Tensor scene_P, //sceneTcw*K: N, L, 3, 4
  int max_displacement,
  int stride){

    const auto N = input1.size(0);
    const auto H = input1.size(1);
    const auto W = input1.size(2);
    const auto C1 = input1.size(3);
    const auto L = input2.size(1);
    const auto SH = input2.size(2);
    const auto SW = input2.size(3);
    const int neighborhood_grid_radius_ = max_displacement;
    const int neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;


    auto input1_grad = torch::zeros({N, H, W, C1},torch::device(torch::kCUDA));
    
    auto input2_grad = torch::zeros({N, L, SH, SW, C1},torch::device(torch::kCUDA));
    
    
    int shared_memory_per_block = C1;
    
    dim3 totalBlocksCorr(W, H, C1);
    dim3 threadsPerBlock(THREADS_PER_WARP);

    for(int n=0;n<N;n++){
      AT_DISPATCH_FLOATING_TYPES(input1.type(), "CorrelateDatabackward_ongpu", ([&] {
        CorrelateDataBackward<scalar_t><<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(scalar_t)>>>(
            grad_output1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            query_coords.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            scene_P.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input1_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input2_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            neighborhood_grid_radius_,
            neighborhood_grid_width_,
            stride,
            n);
      }));
    };
  // 
  return {input1_grad, input2_grad};

}

