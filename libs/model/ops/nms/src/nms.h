#ifndef _NMS_CUDA
#define _NMS_CUDA
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> nms_on_gpu(torch::Tensor coords_2d_grid, //coords_2d_grid: N, H, W, M, 2
    int topk,
    int max_displacement);

#endif