#include "nms.h"
#include <torch/extension.h>
#include <vector>
#include<iostream>
#include<stdio.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// == Forward
std::vector<torch::Tensor> nms_cuda_forward(torch::Tensor coords_2d_grid, //query t: N, H, W, M, 2
                      int topk,
                      int max_displacement){
    CHECK_INPUT(coords_2d_grid);
    

    return nms_on_gpu(coords_2d_grid, topk, max_displacement);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_forward", &nms_cuda_forward, "corr forward (CUDA)");
}