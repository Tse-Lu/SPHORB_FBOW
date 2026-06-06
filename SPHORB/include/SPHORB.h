/*
	AUTHOR:
	Qiang Zhao, email: qiangzhao@tju.edu.cn
	Copyright (C) 2015 Tianjin University
	School of Computer Software
	School of Computer Science and Technology

	LICENSE:
	SPHORB is distributed under the GNU General Public License.  For information on 
	commercial licensing, please contact the authors at the contact address below.

	REFERENCE:
	@article{zhao-SPHORB,
	author   = {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
	title    = {SPHORB: A Fast and Robust Binary Feature on the Sphere},
	journal  = {International Journal of Computer Vision},
	year     = {2015},
	volume   = {113},
	number   = {2},
	pages    = {143-159},
	}
*/

#ifndef _SPHORB_H
#define _SPHORB_H

#include <opencv2/opencv.hpp>	
#include <vector>
#include <stdio.h>
using namespace cv;

namespace cv
{
	class CV_EXPORTS SPHORB : public cv::Feature2D
	{
	public:
		enum { kBytes = 32, SFAST_EDGE = 3, SPHORB_EDGE = 15};

		explicit SPHORB(const std::string &path, int nfeatures = 500, int nlevels = 7, int b=20);
		~SPHORB();

		// returns the descriptor size in bytes
		int descriptorSize();
		// returns the descriptor type
		int descriptorType();

		// Compute the ORB features and descriptors on an image
		void operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints);
		void operator()( InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false );

		std::vector<float*> geoinfos;
		std::vector<cv::Mat> maskes;
		std::vector<std::vector<float*> > imgInfos;
		int levels;

		std::string SPHORB_FOLDER_PATH;

		void initSORB(const std::string &inputPath);
		void uninitSORB();
		void splitSphere2(const cv::Mat& im, cv::Mat& oim, int idx, const float* imgInfo);
		void extendTopRight(cv::Mat& newPart1, const cv::Mat& part2, int edge);
		void extendBottomLeft(cv::Mat& newPart1, const cv::Mat& part2, int edge);
		void extendEdge(cv::Mat& part0, cv::Mat& part1, cv::Mat& part2, cv::Mat& part3, cv::Mat& part4, int edge);
		float inherentAngle(const float* center, const float* axisx);
		float IC_Angle(const cv::Mat& img, const int half_k, cv::Point2f pt, const float* geoinfo);
		void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& kps, int halfPatchSize, const float* geoinfo);
		void computeOrbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, const cv::Point* pattern, uchar* desc, int dsize);
		void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints , cv::Mat& descriptors,
			const std::vector<cv::Point>& pattern, int dsize, int WTA_K);
		void mappingKeypoint(const cv::Mat& img, std::vector<cv::KeyPoint>& kps, int edge, const float* geoinfo, int level);



	protected:
		int barrier;
		int nfeatures;
		int nlevels;

		void computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors );
		void detectImpl( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask=Mat() );
	};

}

#endif