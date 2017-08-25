#pragma once

#include"CamRelocalizer.h"

namespace CamRelocalizer
{
	CamRelocalizer::CamRelocalizer::CamRelocalizer(const std::string & forestPath, const std::string & settingPath)
	{
		// Load settings
		cv::FileStorage settings(settingPath + "/relocalizerSetting.yaml", cv::FileStorage::READ);

		int width = settings["ImageSize.width"];
		int height = settings["ImageSize.height"];

		// Feature generation parameters
		mStride = settings["Stride"];
		mNumOfDepthFeatures = settings["NumOfDepthFeatures"];
		mNumOfRGBFeatures = settings["NumOfRGBFeatures"];

		mDepthOffset.resize(mNumOfDepthFeatures);
		mRGBOffset.resize(mNumOfRGBFeatures);
		mRGBChannels.resize(mNumOfRGBFeatures);

		
		int maxNumOfPixels = width*height / (mStride*mStride);	
		mDescriptors.resize(maxNumOfPixels);
		mSamples.resize(maxNumOfPixels);
		mPredictedData.resize(maxNumOfPixels);
		
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		mRandomGenerator.setSeed(seed);

		generateRandomDepthFeatures();
		generateRandomRGBFeatures();

		// Load trained forest		
		std::string forestFile = forestPath + "/tree.txt";
		mForest = new ForestType(forestFile);
		//mForest->load(treeFile.c_str());
		mNumOfTreesToUse = mForest->getNumOfTrees();//settings["numOfTreesToUse"];

		// Intrinsic Matrix
		float fx = settings["Camera.fx"];
		float fy = settings["Camera.fy"];
		float cx = settings["Camera.cx"];
		float cy = settings["Camera.cy"];
		mIntrinsics << fx, fy, cx, cy;

		// Initialize RansacOptimizer
		int colorThreshold = settings["colorThreshold"];
		float distanceThreshold = settings["minDistance"];
		int maxIterationsGeneration = settings["maxIterationsGenerator"];
		int numOfInitialHypotheses = settings["numOfHypotheses"];
		int numOfNewSamples = settings["numNewSamples"];
		int numOfHypothesesAfterCulling = settings["numInitialSet"];
		int maxIterationsLM = settings["maxIterationsLM"];
		float terminationCriteriaLM = settings["termCriteriaLM"];
		float inlierThreshold = settings["inlierThreshold"];
		
		mRansacOptimizer = new RansacOptimizerType(colorThreshold, distanceThreshold, maxIterationsGeneration, 
			numOfInitialHypotheses, numOfNewSamples, numOfHypothesesAfterCulling, maxIterationsLM, terminationCriteriaLM, 
			inlierThreshold, maxNumOfPixels);

		// Initialize Reservoirs
		mReservoirs = new ReservoirsType();
	}

	CamRelocalizer::~CamRelocalizer()
	{
		mForest->reset();
		mRansacOptimizer->reset();
		mReservoirs->reset();
		delete[] mForest;
		delete[] mRansacOptimizer;
		delete[] mReservoirs;
	}

	void CamRelocalizer::onlineTraining(const cv::Mat & rgbImage, const cv::Mat & depthImage, const Eigen::Matrix4f& cameraPoseInv)
	{
		// Generate samples and descriptors from input images
		generateSamplesAndDescriptors(rgbImage, depthImage, mIntrinsics, cameraPoseInv, mSamples, mDescriptors);
		
		// Predicted leaves indices of the samples
		mForest->predict(mDescriptors, mPredictedLeaves);

		// Insert samples to the specific reservoir based on the predicted leaf indices
		mReservoirs->InsertSamples(mSamples, mPredictedLeaves);

		// Clustering each reservoir
		mReservoirs->findClusters(mReservoirUpdateStartIndex, mNumOfReservoirsToUpdate);

	}

	Eigen::Matrix4f CamRelocalizer::estimatePose(const cv::Mat & rgbImage, const cv::Mat & depthImage)
	{
		Eigen::Matrix4f pose;

		// Compute samples and descriptors from input images
		//computePointsAndDescriptors(rgbImage, depthImage, mIntrinsics, mPoints, mDescriptors);
		generateSamplesAndDescriptors(rgbImage, depthImage, mIntrinsics, Eigen::Matrix4f::Identity(), mSamples, mDescriptors);

		// Predicted leaves indices of the samples
		mForest->predict(mDescriptors, mPredictedLeaves);

		// Extract predicted modes from reservoirs based on the predicted leaves
		mReservoirs->extractModes(mPredictedLeaves, mPredictedData);

		//mPredictedData = mForest->predict(rgbImage, depthImage, mSamples, mNumOfTreesToUse);
		// Run Optimizer;
		pose = mRansacOptimizer->estimateCameraPose(mSamples, mPredictedData);

		return pose;
	}

	void CamRelocalizer::generateSamplesAndDescriptors(const cv::Mat & rgbImgage, const cv::Mat & depthImage, const Eigen::Vector4f & intrinsics, const Eigen::Matrix4f & cameraPose, std::vector<SampleType>& samples, std::vector<DescriptorType>& descriptors)
	{
		int rows = rgbImgage.rows;
		int cols = rgbImgage.cols;

#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (int y = 0; y < rows; y = y + mStride)
		{
			for (int x = 0; x < cols; x = x + mStride)
			{
				float depthValue = depthImage.at<float>(y, x);

				if (depthValue <= 0 || depthValue > 30)
					continue;

				// Compute samples
				SampleType sample;

				const cv::Vec3i& rgbValues = rgbImgage.at<cv::Vec3b>(y, x);
				sample.mColor = Eigen::Vector3i(rgbValues[0], rgbValues[1], rgbValues[2]);
				Eigen::Vector3f cameraspaceCoordinate(((x - intrinsics(2))*depthValue / intrinsics(0)), ((y - intrinsics(3))*depthValue / intrinsics(1)), depthValue);
				sample.mPosition = cameraPose*cameraspaceCoordinate;

#ifdef OPENMP
#pragma omp critical
#endif // OPENMP
				{
					samples.push_back(sample);
				}

				// Compute descriptors
				DescriptorType descriptor;

				for (unsigned featureIndex = 0; featureIndex < mNumOfDepthFeatures; featureIndex++)
				{
					Eigen::Vector2i offset = mDepthOffset[featureIndex];
					int x1 = x + static_cast<int>(static_cast<float>(offset(0)) / depthValue);
					int y1 = y + static_cast<int>(static_cast<float>(offset(1)) / depthValue);

					descriptor.mData[featureIndex] = (depthImage.at<float>(y1, x1) - depthValue)*1000.0f;
				}

				for (unsigned featureIndex = 0; featureIndex < mNumOfRGBFeatures; featureIndex++)
				{
					Eigen::Vector2i offset = mRGBOffset[featureIndex];
					unsigned char channel = mRGBChannels[featureIndex];

					int x1 = x + static_cast<int>(static_cast<float>(offset(0)) / depthValue);
					int y1 = y + static_cast<int>(static_cast<float>(offset(1)) / depthValue);

					descriptor.mData[mNumOfDepthFeatures+featureIndex] = static_cast<float>(rgbImgage.at<cv::Vec3b>(y1, x1)[channel] - rgbValues[channel]);
				}

#ifdef OPENMP
#pragma omp critical
#endif // OPENMP
				{
					descriptors.push_back(descriptor);
				}
			}
		}
	}
	void CamRelocalizer::generateRandomRGBFeatures()
	{
		for (unsigned featureIndex = 0; featureIndex < mNumOfDepthFeatures; featureIndex++)
		{
			mRGBOffset[featureIndex](0)= generateOffset(2, 130);
			mRGBOffset[featureIndex](1) = generateOffset(2, 130);
			mRGBChannels[featureIndex] = mRandomGenerator.uniformIntDistribution(0, 2);
		}
	}
	void CamRelocalizer::generateRandomDepthFeatures()
	{
		for (unsigned featureIndex = 0; featureIndex < mNumOfDepthFeatures; featureIndex++)
		{
			mDepthOffset[featureIndex](0) = generateOffset(1, 65);
			mDepthOffset[featureIndex](1) = generateOffset(1, 65);
		}
	}

	int CamRelocalizer::generateOffset(int a, int b)
	{
		return mRandomGenerator.uniformIntDistribution(a, b) *(mRandomGenerator.uniformIntDistribution(0, 1) * 2 - 1);
	}

}