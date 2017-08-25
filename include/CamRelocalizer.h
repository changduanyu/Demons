#pragma once
#include "RLRforest.h"
#include "RansacOptimizer.h"
#include "CamRelocalizerHelper.h"
#include "RLDataStructure.h"
#include "ForestPredictor.h"
#include "Reservoirs.h"
#include "RandomGenerator.h"

namespace CamRelocalizer
{
	class CamRelocalizer
	{
	public:
		//******************************TYPEDEFINES******************************//
		typedef ForestPredictor<DescriptorType, NodeType> ForestType;
		typedef rlrf::PredictionType<10> PredictionType;
		typedef rlrf::GaussianModeType ModeType;
		typedef RansacOptimizer<HypothesisType, SampleType, PredictionType, ModeType> RansacOptimizerType;
		typedef Reservoirs<SampleType, PredictionType> ReservoirsType;

	public:
		CamRelocalizer(const std::string& forestPath, const std::string& settingPath);
		~CamRelocalizer();

		//*********************************PUBLIC FUNCTIONS*********************************//
		// Online training part given images 
		void onlineTraining(const cv::Mat& rgbImage, const cv::Mat& depthImage, const Eigen::Matrix4f& cameraPoseInv);

		// Estimate camera pose given images
		Eigen::Matrix4f estimatePose(const cv::Mat& rgbImage, const cv::Mat& depthImage);

	private:
		//*********************************PRIVATE VARIABLES*********************************//
		ForestType* mForest;
		unsigned mNumOfTreesToUse;

		RansacOptimizerType* mRansacOptimizer;

		ReservoirsType* mReservoirs;
		// Intrinsic Matrix
		Eigen::Vector4f mIntrinsics;

		// Generated descriptors from images
		std::vector<DescriptorType> mDescriptors;
		// Generated samples from the images
		std::vector<SampleType> mSamples;
		// Predicted Leaves
		std::vector<unsigned> mPredictedLeaves;
		// Predicted data
		std::vector<PredictionType> mPredictedData;

		unsigned mReservoirUpdateStartIndex;
		unsigned mNumOfReservoirsToUpdate;

		RandomGenerator mRandomGenerator;
		//*********************************PRIVATE VARIABLES*********************************//
		// Subsampling ratio
		int mStride;
		int mNumOfRGBFeatures;
		int mNumOfDepthFeatures;

		std::vector<Eigen::Vector2i> mRGBOffset;
		std::vector<unsigned char> mRGBChannels;
		std::vector<Eigen::Vector2i> mDepthOffset;


		//*********************************PRIVATE FUNCTIONS*********************************//
		// Generate samples and descriptors, intrinsics: fx, fy, cx, cy
		void generateSamplesAndDescriptors(const cv::Mat& rgbImgage, const cv::Mat& depthImage, const Eigen::Vector4f& intrinsics, const Eigen::Matrix4f& cameraPose,  std::vector<SampleType>& samples, std::vector<DescriptorType>& descriptors);

		void generateRandomRGBFeatures();
		void generateRandomDepthFeatures();
		int generateOffset(int a, int b);
	};
}