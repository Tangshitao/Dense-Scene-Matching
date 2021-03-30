//
// Created by luwei on 14/01/19.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>

#include "cnn.h"

namespace py = pybind11;

// // Args:
// py::tuple compute_lm_pnp_valid_pixels(
//         py::array_t<float> x_2d,
//         py::array_t<float> X_3d,
//         py::array_t<float> K,
//         py::array_t<int> valid_mask,
//         const int valid_pixels_cnt,
//         float reproj_thres,
//         int obj_hyps,
//         int refine_steps,
//         bool verbose
// )
// {
//     auto start = std::chrono::steady_clock::now();

//     py::buffer_info x_2d_buf = x_2d.request();
//     py::buffer_info X_3d_buf = X_3d.request();
//     py::buffer_info K_buf = K.request();
//     py::buffer_info valid_mask_buf = valid_mask.request();

//     // Check the types
//     if(!py::isinstance<py::array_t<float_t>>(x_2d))
//         throw std::runtime_error("The x_2d array type should be 32-bit float.");
//     if(!py::isinstance<py::array_t<float_t >>(X_3d))
//         throw std::runtime_error("The X_3d array type should be 32-bit float.");
//     if(!py::isinstance<py::array_t<float_t >>(K))
//         throw std::runtime_error("The K array type should be 32-bit float.");

//     // Check the dimension
//     ssize_t H = x_2d.shape(0);          // number of points
//     ssize_t W = x_2d.shape(1);          // number of points

//     if (X_3d.shape(0) != x_2d.shape(0) || X_3d.shape(1) != x_2d.shape(1) )
//         throw std::runtime_error("The dim of (h, w) in x_2d and X_3d should be same.");
//     if (x_2d.shape(2) != 2)
//         throw std::runtime_error("The x_2d is a 2D vector.");
//     if (X_3d.shape(2) != 3)
//         throw std::runtime_error("The X_3d is a 3D vector.");

//     // buffer ptr
//     float *x_2d_buf_ptr = (float*)x_2d_buf.ptr;
//     float *X_3d_buf_ptr = (float*)X_3d_buf.ptr;
//     int *valid_mask_ptr = (int*)valid_mask_buf.ptr;

//     float *valid_pixel_remapping = new float[valid_pixels_cnt*2];
//     memset(valid_pixel_remapping, 0, sizeof(float)*valid_pixels_cnt*2);
//     int process_c = 0;

//     #pragma omp parallel for
//     for(int i = 0; i < H*W; i++){
//         if (valid_mask_ptr[i] == 1)
//         {
//             valid_pixel_remapping[process_c*2] = x_2d_buf_ptr[i*2];
//             valid_pixel_remapping[process_c*2+1] = x_2d_buf_ptr[i*2+1];
//             process_c += 1;     
//         }
//     }

//     cv::Mat_<cv::Point2i> sampling_2d = cv::Mat_<cv::Point2i>(H, W);
//     jp::img_coord_t scene_X_3d = jp::img_coord_t::zeros(H, W);

//     #pragma omp parallel for
//     for (int i = 0; i < H; ++i) {
//         for (int j = 0; j < W; ++j) {
//             size_t offset = i * W + j;

//             // 2D image space coordinate
//             sampling_2d(i, j) = cv::Point2i(int(x_2d_buf_ptr[2 * offset]),
//                     int(x_2d_buf_ptr[2 * offset + 1]));

//             // 3D scene coordinate
//             jp::coord3_t scene_coordinate = jp::coord3_t(
//                     double(X_3d_buf_ptr[3 * offset]),
//                     double(X_3d_buf_ptr[3 * offset + 1]),
//                     double(X_3d_buf_ptr[3 * offset + 2]));

//             scene_X_3d(i, j) = scene_coordinate;
//         }
//     }

//     float *K_buf_ptr = (float*)K_buf.ptr;
//     cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
//     camMat(0, 0) = K_buf_ptr[0];
//     camMat(1, 1) = K_buf_ptr[4];
//     camMat(2, 2) = 1.f;
//     camMat(0, 2) = K_buf_ptr[2];
//     camMat(1, 2) = K_buf_ptr[5];

//     std::vector<jp::cv_trans_t> refHyps;
//     std::vector<double> sfScores;
//     std::vector<std::vector<cv::Point2i>> sampledPoints;
//     std::vector<double> losses;
//     std::vector<cv::Mat_<int>> inlierMaps;
//     double tErr;
//     double rotErr;
//     int hypIdx;

//     double expectedLoss;
//     double sfEntropy;
//     bool correct;

//     // Run the PnP pipeline
//     processImageValidPixels(
//             obj_hyps,
//             camMat,
//             reproj_thres,
//             refine_steps,
//             expectedLoss,
//             sfEntropy,
//             correct,
//             refHyps,
//             sfScores,
//             scene_X_3d,
//             sampling_2d,
//             sampledPoints,
//             valid_pixel_remapping,
//             valid_pixels_cnt,
//             losses,
//             inlierMaps,
//             tErr,
//             rotErr,
//             hypIdx,
//             false, 
//             verbose);


//     auto pred_pose_vec = py::array_t<float_t>(6);
//     py::buffer_info pred_pose_vec_buf = pred_pose_vec.request(true);
//     float *pred_pose_vec_buf_ptr = (float*)pred_pose_vec_buf.ptr;

//     auto inlier_map = py::array_t<int32_t >(H * W);
//     py::buffer_info inlier_map_buf = inlier_map.request(true);
//     int *inlier_map_buf_ptr = (int*)inlier_map_buf.ptr;

//     // assign output
//     pred_pose_vec_buf_ptr[0] = (float) refHyps[hypIdx].first.at<double>(0, 0);     // 5  - selected pose, rotation (1st component of Rodriguez vector)
//     pred_pose_vec_buf_ptr[1] = (float) refHyps[hypIdx].first.at<double>(1, 0);     // 6  - selected pose, rotation (2nd component of Rodriguez vector)
//     pred_pose_vec_buf_ptr[2] = (float) refHyps[hypIdx].first.at<double>(2, 0);     // 7  - selected pose, rotation (3th component of Rodriguez vector)
//     pred_pose_vec_buf_ptr[3] = (float) refHyps[hypIdx].second.at<double>(0, 0);    // 8  - selected pose, translation in m (x)
//     pred_pose_vec_buf_ptr[4] = (float) refHyps[hypIdx].second.at<double>(0, 1);    // 9  - selected pose, translation in m (y)
//     pred_pose_vec_buf_ptr[5] = (float) refHyps[hypIdx].second.at<double>(0, 2);    // 10 - selected pose, translation in m (z)

//     #pragma omp parallel for
//     for (int i = 0; i < H; ++i) {
//         for (int j = 0; j < W; ++j) {
//             size_t offset = i * W + j;
//             inlier_map_buf_ptr[offset] = inlierMaps[hypIdx](i, j);
//         }
//     }

//     delete[] valid_pixel_remapping;
//     pred_pose_vec.resize({6});
//     inlier_map.resize({H, W});

//     py::tuple res = py::make_tuple(pred_pose_vec, inlier_map);
 
//    	auto end = std::chrono::steady_clock::now();
//     if (verbose)
//     	std::cout << "Total time in milliseconds : " 
// 		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
// 		<< " ms" << std::endl;

//     return res;
// }


py::tuple compute_lm_pnp(
        py::array_t<float> x_2d,
        py::array_t<float> X_3d,
        py::array_t<float> K,
        float reproj_thres,
        int obj_hyps,
        int refine_steps, 
        bool verbose) {
  	auto start = std::chrono::steady_clock::now();

    py::buffer_info x_2d_buf = x_2d.request();
    py::buffer_info X_3d_buf = X_3d.request();
    py::buffer_info K_buf = K.request();

    // Check the types
    if(!py::isinstance<py::array_t<float_t>>(x_2d))
        throw std::runtime_error("The x_2d array type should be 32-bit float.");
    if(!py::isinstance<py::array_t<float_t >>(X_3d))
        throw std::runtime_error("The X_3d array type should be 32-bit float.");
    if(!py::isinstance<py::array_t<float_t >>(K))
        throw std::runtime_error("The K array type should be 32-bit float.");

    // Check the dimension
    ssize_t H = x_2d.shape(0);          // number of points
    ssize_t W = x_2d.shape(1);          // number of points

    if (X_3d.shape(0) != x_2d.shape(0) || X_3d.shape(1) != x_2d.shape(1) )
        throw std::runtime_error("The dim of (h, w) in x_2d and X_3d should be same.");
    if (x_2d.shape(2) != 2)
        throw std::runtime_error("The x_2d is a 2D vector.");
    if (X_3d.shape(2) != 3)
        throw std::runtime_error("The X_3d is a 3D vector.");

    // buffer ptr
    float *x_2d_buf_ptr = (float*)x_2d_buf.ptr;
    float *X_3d_buf_ptr = (float*)X_3d_buf.ptr;

    cv::Mat_<cv::Point2i> sampling_2d = cv::Mat_<cv::Point2i>(H, W);
    jp::img_coord_t scene_X_3d = jp::img_coord_t::zeros(H, W);

    #pragma omp parallel for
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            size_t offset = i * W + j;

            // 2D image space coordinate
            sampling_2d(i, j) = cv::Point2i(int(x_2d_buf_ptr[2 * offset]),
                    int(x_2d_buf_ptr[2 * offset + 1]));

            // 3D scene coordinate
            jp::coord3_t scene_coordinate = jp::coord3_t(
                    double(X_3d_buf_ptr[3 * offset]),
                    double(X_3d_buf_ptr[3 * offset + 1]),
                    double(X_3d_buf_ptr[3 * offset + 2]));

            scene_X_3d(i, j) = scene_coordinate;
        }
    }

    float *K_buf_ptr = (float*)K_buf.ptr;
    cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
    camMat(0, 0) = K_buf_ptr[0];
    camMat(1, 1) = K_buf_ptr[4];
    camMat(2, 2) = 1.f;
    camMat(0, 2) = K_buf_ptr[2];
    camMat(1, 2) = K_buf_ptr[5];

    std::vector<jp::cv_trans_t> refHyps;
    std::vector<double> sfScores;
    std::vector<std::vector<cv::Point2i>> sampledPoints;
    std::vector<double> losses;
    std::vector<cv::Mat_<int>> inlierMaps;
    double tErr;
    double rotErr;
    int hypIdx;

    double expectedLoss;
    double sfEntropy;
    bool correct;

    // Run the PnP pipeline
    processImageModified(
            obj_hyps,
            camMat,
            reproj_thres,
            refine_steps,
            expectedLoss,
            sfEntropy,
            correct,
            refHyps,
            sfScores,
            scene_X_3d,
            sampling_2d,
            sampledPoints,
            losses,
            inlierMaps,
            tErr,
            rotErr,
            hypIdx,
            false, 
            verbose);

    // output variables
//    auto pred_pose = py::array_t<float_t>(3*4);
//    py::buffer_info pred_pose_buf = pred_pose.request(true);
//    float *pred_pose_buf_ptr = (float*)pred_pose_buf.ptr;

    auto pred_pose_vec = py::array_t<float_t>(6);
    py::buffer_info pred_pose_vec_buf = pred_pose_vec.request(true);
    float *pred_pose_vec_buf_ptr = (float*)pred_pose_vec_buf.ptr;

    auto inlier_map = py::array_t<int32_t >(H * W);
    py::buffer_info inlier_map_buf = inlier_map.request(true);
    int *inlier_map_buf_ptr = (int*)inlier_map_buf.ptr;

    // assign output
    pred_pose_vec_buf_ptr[0] = (float) refHyps[hypIdx].first.at<double>(0, 0);     // 5  - selected pose, rotation (1st component of Rodriguez vector)
    pred_pose_vec_buf_ptr[1] = (float) refHyps[hypIdx].first.at<double>(1, 0);     // 6  - selected pose, rotation (2nd component of Rodriguez vector)
    pred_pose_vec_buf_ptr[2] = (float) refHyps[hypIdx].first.at<double>(2, 0);     // 7  - selected pose, rotation (3th component of Rodriguez vector)
    pred_pose_vec_buf_ptr[3] = (float) refHyps[hypIdx].second.at<double>(0, 0);    // 8  - selected pose, translation in m (x)
    pred_pose_vec_buf_ptr[4] = (float) refHyps[hypIdx].second.at<double>(0, 1);    // 9  - selected pose, translation in m (y)
    pred_pose_vec_buf_ptr[5] = (float) refHyps[hypIdx].second.at<double>(0, 2);    // 10 - selected pose, translation in m (z)

    #pragma omp parallel for
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            size_t offset = i * W + j;
            inlier_map_buf_ptr[offset] = inlierMaps[hypIdx](i, j);
        }
    }

    pred_pose_vec.resize({6});
    inlier_map.resize({H, W});

    py::tuple res = py::make_tuple(pred_pose_vec, inlier_map);

  	auto end = std::chrono::steady_clock::now();
    if (verbose)
    	std::cout << "Total time in milliseconds : " 
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " ms" << std::endl;

    return res;
}

PYBIND11_MODULE(lm_pnp, m){
    m.doc() = "Python binding of LessMore PnP.";
    m.def("compute_lm_pnp", &compute_lm_pnp, "Check the input array");
    // m.def("compute_lm_pnp_valid_pixels", &compute_lm_pnp_valid_pixels, "compute the pnp with valid pixels");
}
