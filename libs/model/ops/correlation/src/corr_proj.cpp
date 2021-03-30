#include "corr_proj.h"
#include <torch/extension.h>
#include <vector>
#include<iostream>
#include<stdio.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// == Forward
std::vector<torch::Tensor> corr_cuda_forward(torch::Tensor input1, //query t: N, H, W, C1
                      torch::Tensor input2, //scene : N, H, W, C1
                      torch::Tensor query_coords, // scene coords: N, H, W, 3
                      torch::Tensor scene_coords, // scene coords: N, H, W, 3
                      torch::Tensor scene_P, //sceneTcw*K, N, 3, 4
                      int max_displacement,
                      int stride){
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    CHECK_INPUT(query_coords);
    CHECK_INPUT(scene_coords);
    CHECK_INPUT(scene_P);

    return CorrelateData_ongpu(input1, input2, query_coords, scene_coords, scene_P, max_displacement, stride);

}

std::vector<torch::Tensor> corr_cuda_backward(torch::Tensor grad_output1, //query t: N, H, W, C1
                      torch::Tensor input1, //scene : N, H, W, C1
                      torch::Tensor input2, // scene coords: N, H, W, 3
                      torch::Tensor query_coords, // scene coords: N, H, W, 3
                      torch::Tensor scene_P, //sceneTcw*K, N, 3, 4
                      int max_displacement,
                      int stride){
    CHECK_INPUT(grad_output1);
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    CHECK_INPUT(query_coords);
    CHECK_INPUT(scene_P);

    return CorrelateData_backward_ongpu(grad_output1, input1, input2, query_coords, scene_P, max_displacement, stride);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("corr_proj_forward", &corr_cuda_forward, "corr forward (CUDA)");
  m.def("corr_proj_backward", &corr_cuda_backward, "corr forward (CUDA)");
}