#ifndef _CORR_CUDA
#define _CORR_CUDA
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> corr_cuda_forward(torch::Tensor input1,       //query t: N, H, W, C1
                                             torch::Tensor input2,       //scene : N, H, W, C1
                                             torch::Tensor query_coords, // scene coords: N, H, W, 3
                                             torch::Tensor scene_coords, // scene coords: N, H, W, 3
                                             torch::Tensor scene_P,      //sceneTcw*K, N, 3, 4
                                             int max_displacement,
                                             int stride);

std::vector<torch::Tensor> CorrelateData_ongpu(torch::Tensor input1,       //query t: N, H, W, C1
                                               torch::Tensor input2,       //scene : N, H, W, C1
                                               torch::Tensor query_coords, // scene coords: N, H, W, 3
                                               torch::Tensor scene_coords, // scene coords: N, H, W, 3
                                               torch::Tensor scene_P,      //sceneTcw*K, N, 3, 4
                                               int max_displacement,
                                               int stride);

std::vector<torch::Tensor> CorrelateData_backward_ongpu(torch::Tensor grad_output1, //query t: N, H, W, C1
                                                        torch::Tensor input1,       //scene : N, L, H, W, C1
                                                        torch::Tensor input2,       // scene coords: N, 3, H, W
                                                        torch::Tensor query_coords, // scene coords: N, L, 3, H, W
                                                        torch::Tensor scene_P,      //sceneTcw*K: N, L, 3, 4
                                                        int max_displacement,
                                                        int stride);

#endif