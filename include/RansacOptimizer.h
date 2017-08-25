#pragma once

#include<vector>
#include<algorithm>
#include<chrono>
#include<math.h>

#include "RandomGenerator.h"
#include "RansacHelper.h"

namespace CamRelocalizer
{
	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	class RansacOptimizer
	{
	public:
		RansacOptimizer(int colorThreshold, float distanceThreshold, int maxNumOfIters, int numOfInitialHypothesis, int numOfNewSamples, int numOfHypothesisAfterCulling, int maxNumOfItersLM, float terminationCriteria, float inlierThreshold, int maxNumOfPoints);

		//~RansacOptimizer() {};

		//*********************PUBLIC FUNCTIONS*************************//
		// Estimate Camera Pose from given data
		Eigen::Matrix4f estimateCameraPose(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData);

		void reset();

	private:
		//*********************PRIVATE VARIABLES*************************//
		// Used Parameters when generating pose hypothesis 
		int mColorThreshold;
		float mDistanceThreshold;
		int mMaxNumOfIters;
		int mNumOfInitialHypothesis;

		// Used Parameters during Ransac
		int mNumOfNewSamples;
		int mNumOfHypothesisAfterCulling;
		int mMaxNumOfItersLM;

		// Generated pose hypothesis
		std::vector<HypothesisType> mPoseHypotheses;
		// Selected point indices
		std::vector<int> mSelectedPointIndices;
		// New sampled point indices
		std::vector<int> mNewPointIndices;
		// Camera space coordinates inliers for optimization. size:mNewPointIndices.size() * mPoseHypotheses.size()
		std::vector<Eigen::Vector4f> mXcsInliers;
		// Selected best modes. size:mNewPointIndices.size() * mPoseHypotheses.size()
		std::vector<ModeType> mBestModes;

		// Inlier threshold
		float mInlierThreshold;
		// Termination criteria for LM
		float mTerminationCriteriaLM;
		// Random number generator
		std::vector<RandomGenerator> mRandomGenerators;

		//*********************PRIVATE FUNCTIONS*************************//
		void generatePoseHypotheses(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData);

		void sampleNewPoints(int pointsSize);

		void computeEnergies(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData, bool isSelectModes);

		void optimizeHypotheses(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData);

		//*********************Private Functions*************************//
		void runKabschAlgorithm();

		void computeOneEnergy(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData, int hIndex, bool isSelectInliers);

		void optimizeOneHypothesis(int hIndex, const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData);
	};

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::RansacOptimizer(int colorThreshold, float distanceThreshold, int maxNumOfIters, int numOfInitialHypothesis, int numOfNewSamples, int numOfHypothesisAfterCulling, int maxNumOfItersLM, float terminationCriteria, float inlierThreshold, int maxNumOfPoints)
		: mColorThreshold(colorThreshold), mDistanceThreshold(distanceThreshold), mMaxNumOfIters(maxNumOfIters),
		mNumOfInitialHypothesis(numOfInitialHypothesis), mNumOfNewSamples(numOfNewSamples), mNumOfHypothesisAfterCulling(numOfHypothesisAfterCulling),
		mMaxNumOfItersLM(maxNumOfItersLM), mInlierThreshold(inlierThreshold), mTerminationCriteriaLM(terminationCriteria)
	{
		mPoseHypotheses.reserve(mNumOfInitialHypothesis);
		mSelectedPointIndices.reserve(maxNumOfPoints);
		mNewPointIndices.reserve(numOfNewSamples);

		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		mRandomGenerators.resize(numOfInitialHypothesis);
		for (int i = 0; i < numOfInitialHypothesis; i++)
		{
			mRandomGenerators[i].setSeed(seed + i);
		}
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	Eigen::Matrix4f RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::estimateCameraPose(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData)
	{
		generatePoseHypotheses(points, predictedData);

		// Reset selected point indices
		size_t pointsSize = points.size();
		mSelectedPointIndices.clear();
		mSelectedPointIndices.resize(pointsSize);

		if (mNumOfHypothesisAfterCulling < mPoseHypotheses.size())
		{
			sampleNewPoints(pointsSize);

			computeEnergies(points, predictedData, false);

			mPoseHypotheses.resize(mNumOfHypothesisAfterCulling);
		}

		if (mPoseHypotheses.size() == 1)
		{
			sampleNewPoints(pointsSize);
			optimizeHypotheses(points, predictedData);
		}

		//sampleNewPoints(pointsSize);
		computeEnergies(points, predictedData, true);
		while (mPoseHypotheses.size() > 1)
		{
			optimizeHypotheses(points, predictedData);

			computeEnergies(points, predictedData, true);

			sampleNewPoints(pointsSize);
			mPoseHypotheses.resize(mPoseHypotheses.size() / 2);
		}

		return mPoseHypotheses.size()>0 ? mPoseHypotheses[0].mPose : Eigen::Matrix4f::Identity();
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	inline void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::reset()
	{
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::generatePoseHypotheses(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData)
	{
		mPoseHypotheses.clear();
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (int i = 0; i < mNumOfInitialHypothesis; i++)
		{
			HypothesisType currHypothesis;
			bool isValid = generateOneHypothesis(points, predictedData, currHypothesis, mMaxNumOfIters, true, 30, 0.3f, 0.025f, mRandomGenerators[i]);

			if (isValid)
			{
#ifdef OPENMP
#pragma omp critical
#endif // OPENMP
				{
					mPoseHypotheses.push_back(currHypothesis);
				}
			}
		}

		runKabschAlgorithm();
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::sampleNewPoints(int pointsSize)
	{
		mNewPointIndices.clear();
#ifdef OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < mNumOfNewSamples; i++)
		{
			int randInt = sampleOnePoints(mRandomGenerators[i], mSelectedPointIndices, pointsSize);

			if (randInt >= 0)
#ifdef OPENMP
#pragma omp automic capture
#endif // OPENMP
				mNewPointIndices.push_back(randInt);
		}
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::computeEnergies(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData, bool isSelectModes)
	{
		size_t numOfHypotheses = mPoseHypotheses.size();
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		{
			for (size_t i = 0; i < numOfHypotheses; i++)
			{
				computeOneEnergy(points, predictedData, static_cast<int>(i), isSelectModes);
			}

		}
		std::sort(mPoseHypotheses.begin(), mPoseHypotheses.end());
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::optimizeHypotheses(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData)
	{
		size_t numOfHypotheeses = mPoseHypotheses.size();
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (size_t hIndex = 0; hIndex < numOfHypotheeses; hIndex++)
		{
			optimizeOneHypothesis(hIndex, points, predictedData);
		}
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::runKabschAlgorithm()
	{
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (size_t i = 0; i < mPoseHypotheses.size(); i++)
		{
			HypothesisType& currHypothesis = mPoseHypotheses[i];
			const Eigen::Vector3f* xws = currHypothesis.mXws;
			const Eigen::Vector3f* xcs = currHypothesis.mXcs;

			currHypothesis.mPose = kabschAlgorithm(xcs, xws);
		}
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::computeOneEnergy(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData, int hIndex, bool isSelectInliers)
	{

		HypothesisType& currHypothesis = mPoseHypotheses[hIndex];

		float energy = computeEnergy(currHypothesis, points, predictedData, mNewPointIndices, mInlierThreshold, isSelectInliers);

		currHypothesis.mEnergy = energy;
	}

	template<typename HypothesisType, typename PointType, typename PredictionType, typename ModeType>
	void RansacOptimizer<HypothesisType, PointType, PredictionType, ModeType>::optimizeOneHypothesis(int hIndex, const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData)
	{
		HypothesisType& currHypothesis = mPoseHypotheses[hIndex];
		const int sampledPointSize = static_cast<int>(mNewPointIndices.size());

		optimizationLM(currHypothesis, points, predictedData);
	}

}