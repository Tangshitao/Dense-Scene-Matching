/*
Copyright (c) 2016, TU Dresden
Copyright (c) 2017, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#define CNN_OBJ_MAXINPUT 100.0 // reprojection errors are clamped at this magnitude

#include "util.h"
#include "maxloss.h"

#include <math.h>


double sigmoid(double x)
{
    double exp_value;
    double return_value;

    /*** Exponential calculation ***/
    exp_value = exp(-x);

    /*** Final sigmoid value ***/
    return_value = 1 / (1 + exp_value);

    return return_value;
}

/**
 * @brief Wrapper around the OpenCV PnP function that returns a zero pose in case PnP fails. See also documentation of cv::solvePnP.
 * @param objPts List of 3D points.
 * @param imgPts Corresponding 2D points.
 * @param camMat Calibration matrix of the camera.
 * @param distCoeffs Distortion coefficients.
 * @param rot Output parameter. Camera rotation.
 * @param trans Output parameter. Camera translation.
 * @param extrinsicGuess If true uses input rot and trans as initialization.
 * @param methodFlag Specifies the PnP algorithm to be used.
 * @return True if PnP succeeds.
 */
inline bool safeSolvePnP(
    const std::vector<cv::Point3f>& objPts,
    const std::vector<cv::Point2f>& imgPts,
    const cv::Mat& camMat,
    const cv::Mat& distCoeffs,
    cv::Mat& rot,
    cv::Mat& trans,
    bool extrinsicGuess,
    int methodFlag)
{
    if(rot.type() == 0) rot = cv::Mat_<double>::zeros(1, 3);
    if(trans.type() == 0) trans= cv::Mat_<double>::zeros(1, 3);

    if(!cv::solvePnP(objPts, imgPts, camMat, distCoeffs, rot, trans, extrinsicGuess,methodFlag))
    {
        rot = cv::Mat_<double>::zeros(1, 3);
        trans = cv::Mat_<double>::zeros(1, 3);
        return false;
    }
    return true;
}

/**
 * @brief Calculate the Shannon entropy of a discrete distribution.
 * @param dist Discrete distribution. Probability per entry, should sum to 1.
 * @return  Shannon entropy.
 */
double entropy(const std::vector<double>& dist)
{
    double e = 0;
    for(unsigned i = 0; i < dist.size(); i++)
	if(dist[i] > 0)
	    e -= dist[i] * std::log2(dist[i]);
    
    return e;
}

/**
 * @brief Draws an entry of a discrete distribution according to the given probabilities.
 *
 * If randomDraw is false in the properties, this function will return the entry with the max. probability.
 *
 * @param probs Discrete distribution. Probability per entry, should sum to 1.
 * @return Chosen entry.
 */
int draw(const std::vector<double>& probs)
{
    std::map<double, int> cumProb;
    double probSum = 0;
    double maxProb = -1;
    double maxIdx = 0; 
    
    for(unsigned idx = 0; idx < probs.size(); idx++)
    {
	if(probs[idx] < EPS) continue;
	
	probSum += probs[idx];
	cumProb[probSum] = idx;
	
	if(maxProb < 0 || probs[idx] > maxProb)
	{
	    maxProb = probs[idx];
	    maxIdx = idx;
	}
    }

    return maxIdx;
}

/**
 * @brief Calculate an image of reprojection errors for the given object coordinate prediction and the given pose.
 * @param hyp Pose estimate.
 * @param objectCoordinates Object coordinate estimate.
 * @param sampling Subsampling of the input image.
 * @param camMat Calibration matrix of the camera.
 * @return Image of reprojectiob errors.
 */
cv::Mat_<float> getDiffMap(
  const jp::cv_trans_t& hyp,
  const jp::img_coord_t& objectCoordinates,
  const cv::Mat_<cv::Point2i>& sampling,
  const cv::Mat& camMat)
{
    cv::Mat_<float> diffMap(sampling.size());
  
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> projections;	
    std::vector<cv::Point2f> points2D;
    std::vector<cv::Point2f> sources2D;
    
    // collect 2D-3D correspondences
    for(unsigned x = 0; x < sampling.cols; x++)
    for(unsigned y = 0; y < sampling.rows; y++)
    {
        // get 2D location of the original RGB frame
	cv::Point2f pt2D(sampling(y, x).x, sampling(y, x).y);
	
        // get associated 3D object coordinate prediction
	points3D.push_back(cv::Point3f(
	    objectCoordinates(y, x)(0), 
	    objectCoordinates(y, x)(1), 
	    objectCoordinates(y, x)(2)));
	points2D.push_back(pt2D);
	sources2D.push_back(cv::Point2f(x, y));
    }
    
    if(points3D.empty()) return diffMap;

    // project object coordinate into the image using the given pose
    cv::projectPoints(points3D, hyp.first, hyp.second, camMat, cv::Mat(), projections);
    
    // measure reprojection errors
    for(unsigned p = 0; p < projections.size(); p++)
    {
	cv::Point2f curPt = points2D[p] - projections[p];
	float l = std::min(cv::norm(curPt), CNN_OBJ_MAXINPUT);
	diffMap(sources2D[p].y, sources2D[p].x) = l;
    }

    return diffMap;    
}

/**
 * @brief Applies soft max to the given list of scores.
 * @param scores List of scores.
 * @return Soft max distribution (sums to 1)
 */
std::vector<double> softMax(const std::vector<double>& scores)
{
    double maxScore = 0;
    for(unsigned i = 0; i < scores.size(); i++)
        if(i == 0 || scores[i] > maxScore) maxScore = scores[i];
	
    std::vector<double> sf(scores.size());
    double sum = 0.0;
    
    for(unsigned i = 0; i < scores.size(); i++)
    {
	sf[i] = std::exp(scores[i] - maxScore);
	sum += sf[i];
    }
    for(unsigned i = 0; i < scores.size(); i++)
    {
	sf[i] /= sum;
// 	std::cout << "score: " << scores[i] << ", prob: " << sf[i] << std::endl;
    }
    
    return sf;
}

/**
 * Compute the scores of each pose hypothesis using soft inlier count (as in "Learing Less is More")
 *
 * @param diffMaps List of images of reprojection errors.
 */
std::vector<double> getHypScores(std::vector<cv::Mat_<float>> & diffMaps, double inlierThresh)
{
//    inlierThresh = 10.0;
    double inlierSoft = 50.0;  // sigmoid softness (beta)
    std::vector<double> scores;
    for(int i = 0; i < diffMaps.size(); i++)
    {
        double cur_score = 0.0;
        for(unsigned x = 0; x < diffMaps[i].cols; x++)
            for(unsigned y = 0; y < diffMaps[i].rows; y++)
//                cur_score += sigmoid(inlierThresh - inlierSoft * diffMaps[i](y, x));
                cur_score += sigmoid(inlierSoft * (inlierThresh - diffMaps[i](y, x)));
        scores.push_back(cur_score);
    }
    return scores;
}


/**
 * @brief Processes a frame, ie. takes object coordinates, estimates poses, selects the best one and measures the error.
 *
 * This function performs the forward pass of DSAC but also calculates many intermediate results
 * for the backward pass (ie it can be made faster if one cares only about the forward pass).
 *
 * @param poseGT Ground truth pose (for evaluation only).
 * @param stateObj Lua state for access to the score CNN.
 * @param objHyps Number of hypotheses to be drawn.
 * @param camMat Calibration parameters of the camera.
 * @param inlierThreshold2D Inlier threshold in pixels.
 * @param refSteps Max. refinement steps (iterations).
 * @param expectedLoss Output paramter. Expectation of loss of the discrete hypothesis distributions.
 * @param sfEntropy Output parameter. Shannon entropy of the soft max distribution of hypotheses.
 * @param correct Output parameter. Was the final, selected hypothesis correct?
 * @param refHyps Output parameter. List of refined hypotheses sampled for the given image.
 * @param sfScores Output parameter. Soft max distribution for the sampled hypotheses.
 * @param estObj Output parameter. Estimated object coordinates (subsampling of the complete image).
 * @param sampling Output parameter. Subsampling of the RGB image.
 * @param sampledPoints Output parameter. List of initial 2D pixel locations of the subsampled input RGB image. 4 pixels per hypothesis.
 * @param losses Output parameter. List of losses of the sampled hypotheses.
 * @param inlierMaps Output parameter. Maps indicating which pixels of the subsampled input image have been inliers in the last step of hypothesis refinement, one map per hypothesis.
 * @param tErr Output parameter. Translational (in m) error of the final, selected hypothesis.
 * @param rotErr Output parameter. Rotational error of the final, selected hypothesis.
 * @param hypIdx Output parameter. Index of the final, selected hypothesis.
 * @param training True if training mode. Controls whether all hypotheses are refined or just the selected one.
 */
void processImageModified(
        int objHyps,
        const cv::Mat& camMat,
        float inlierThreshold2D,
        int refSteps,
        double& expectedLoss,
        double& sfEntropy,
        bool& correct,
        std::vector<jp::cv_trans_t>& refHyps,
        std::vector<double>& sfScores,
        const jp::img_coord_t& estObj,
        const cv::Mat_<cv::Point2i>& sampling,
        std::vector<std::vector<cv::Point2i>>& sampledPoints,
        std::vector<double>& losses,
        std::vector<cv::Mat_<int>>& inlierMaps,
        double& tErr,
        double& rotErr,
        int& hypIdx,
        bool training = false, bool verbose=false)
{
    std::cout << "Sampling " << objHyps << " hypotheses." << std::endl;

    sampledPoints.resize(objHyps);    // keep track of the points each hypothesis is sampled from
    refHyps.resize(objHyps);
    std::vector<std::vector<cv::Point2f>> imgPts(objHyps);
    std::vector<std::vector<cv::Point3f>> objPts(objHyps);

    // sample hypotheses
//    #pragma omp parallel for
    for(unsigned h = 0; h < refHyps.size(); h++)
        while(true)
        {
            std::vector<cv::Point2f> projections;
            cv::Mat_<uchar> alreadyChosen = cv::Mat_<uchar>::zeros(estObj.size());
            imgPts[h].clear();
            objPts[h].clear();
            sampledPoints[h].clear();

            for(int j = 0; j < 4; j++)
            {
                // 2D location in the subsampled image
                int x = rand() % estObj.cols;//irand(0, estObj.cols);
                int y = rand() % estObj.rows;//irand(0, estObj.rows);

                if(alreadyChosen(y, x) > 0)
                {
                    j--;
                    continue;
                }

                alreadyChosen(y, x) = 1;

                imgPts[h].push_back(sampling(y, x)); // 2D location in the original RGB image
                objPts[h].push_back(cv::Point3f(estObj(y, x))); // 3D object coordinate
                sampledPoints[h].push_back(cv::Point2i(x, y)); // 2D pixel location in the subsampled image
            }

            if(!safeSolvePnP(objPts[h], imgPts[h], camMat, cv::Mat(), refHyps[h].first, refHyps[h].second, false, CV_P3P))
            {
                continue;
            }

            cv::projectPoints(objPts[h], refHyps[h].first, refHyps[h].second, camMat, cv::Mat(), projections);

            // check reconstruction, 4 sampled points should be reconstructed perfectly
            bool foundOutlier = false;
            for(unsigned j = 0; j < imgPts[h].size(); j++)
            {
                if(cv::norm(imgPts[h][j] - projections[j]) < inlierThreshold2D)
                    continue;
                foundOutlier = true;
                break;
            }
            if(foundOutlier)
                continue;
            else
                break;
        }

    std::cout << "Calculating scores." << std::endl;

    // compute reprojection error images
    std::vector<cv::Mat_<float>> diffMaps(objHyps);
//    #pragma omp parallel for
    for(unsigned h = 0; h < refHyps.size(); h++)
        diffMaps[h] = getDiffMap(refHyps[h], estObj, sampling, camMat);

    // compute hypothesis scores
    std::vector<double> scores = getHypScores(diffMaps, inlierThreshold2D);

    std::cout << scores[0] << ' ' << scores[1] << ' ' << scores[2] << std::endl;

    std::cout << "Drawing final Hypothesis." << std::endl;

    // apply soft max to scores to get a distribution
    sfScores = softMax(scores);
    sfEntropy = entropy(sfScores); // measure distribution entropy
    hypIdx = draw(sfScores); // select winning hypothesis

    std::cout << "Refining poses:" << std::endl;

    // collect inliers
    inlierMaps.resize(refHyps.size());

    double convergenceThresh = 0.01; // stop refinement if 6D pose vector converges

//    #pragma omp parallel for
    for(unsigned h = 0; h < refHyps.size(); h++)
    {
        if(!training && hypIdx != h)
            continue; // in test mode only refine selected hypothesis

        cv::Mat_<float> localDiffMap = diffMaps[h];

        // refine current hypothesis
        for(unsigned rStep = 0; rStep < refSteps; rStep++)
        {
            // collect inliers
            std::vector<cv::Point2f> localImgPts;
            std::vector<cv::Point3f> localObjPts;
            cv::Mat_<int> localInlierMap = cv::Mat_<int>::zeros(diffMaps[h].size());

            for(unsigned x = 0; x < localDiffMap.cols; x++)
                for(unsigned y = 0; y < localDiffMap.rows; y++)
                {
                    if(localDiffMap(y, x) < inlierThreshold2D)
                    {
                        localImgPts.push_back(sampling(y, x));
                        localObjPts.push_back(cv::Point3f(estObj(y, x)));
                        localInlierMap(y, x) = 1;
                    }
                }

            if(localImgPts.size() < 4)
                break;

            // recalculate pose
            jp::cv_trans_t hypUpdate;
            hypUpdate.first = refHyps[h].first.clone();
            hypUpdate.second = refHyps[h].second.clone();

            if(!safeSolvePnP(localObjPts, localImgPts, camMat, cv::Mat(), hypUpdate.first, hypUpdate.second, true, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
                break; //abort if PnP fails

            if(maxLoss(hypUpdate, refHyps[h]) < convergenceThresh)
                break; // convergned

            refHyps[h] = hypUpdate;
            inlierMaps[h] = localInlierMap;

            // recalculate pose errors
            localDiffMap = getDiffMap(refHyps[h], estObj, sampling, camMat);
        }
    }

//    std::cout << "Final Result:" << std::endl;
//
//    // evaluated poses
//    expectedLoss = expectedMaxLoss(hypGT, refHyps, sfScores, losses);
//    std::cout << "Loss of winning hyp: " << maxLoss(hypGT, refHyps[hypIdx]) << ", prob: " << sfScores[hypIdx] << ", expected loss: " << expectedLoss << std::endl;
//
//    // we measure error of inverted poses (because we estimate scene poses, not camera poses)
//    jp::cv_trans_t invHypGT = getInvHyp(hypGT);
//    jp::cv_trans_t invHypEst = getInvHyp(refHyps[hypIdx]);
//
//    rotErr = calcAngularDistance(invHypGT, invHypEst);
//    tErr = cv::norm(invHypEst.second - invHypGT.second);
//
//    correct = false;
//    if(rotErr < 5 && tErr < 0.05)
//    {
//        std::cout << GREENTEXT("Rotation Err: " << rotErr << "deg, Translation Err: " << tErr * 100 << "cm") << std::endl << std::endl;
//        correct = true;
//    }
//    else
//        std::cout << REDTEXT("Rotation Err: " << rotErr << "deg, Translation Err: " << tErr * 100 << "cm") << std::endl << std::endl;

}


