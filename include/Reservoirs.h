#pragma once
#include <algorithm>
#include <Eigen\Dense>

#include "RandomGenerator.h"
//#include "ReservoirsHelper.h"

namespace CamRelocalizer
{
	template<typename SampleType, typename PredictionType>
	class Reservoirs
	{
	public:
		struct ClusterIDSize
		{
			unsigned mClusterId;
			unsigned mSize;
		};
	public:
		Reservoirs(unsigned numOfTrees, unsigned numOfReservoirs, unsigned maxNumOfSamplesPerReservoir, float mSigma,
			float mTau, unsigned maxNumOfClustersPerReservoir, unsigned minNumOfSamplesPerCluster);

		//**************************************PUBLIC FUNCTIONS**************************************//
		// Insert samples to the reservoirs based on found leaf indices
		void InsertSamples(const std::vector<SampleType>& samples, const std::vector<unsigned>& reservoirIndices);

		// Finding clusters in reservoirs using really quick shift algorithm
		void findClusters(unsigned updateStartIndex, unsigned numOfUpdatedLeaves);

		// Predict the samples
		void extractModes(const std::vector<unsigned>& reservoirIndices, std::vector<PredictionType>& predictions );

		// Compare function
		static bool comparision(const ClusterIDSize& c1, const ClusterIDSize& c2);
		void reset();

	private:
		//**************************************PRIVATE VARIABLES**************************************//
		// Reservoirs that stores samples
		std::vector<SampleType> mReservoirs;
		// Number of samples in a reservoir
		std::vector<unsigned> mNumOfSamplesInReservoirs;
		//  Number of insertions in a reservoir
		std::vector<unsigned> mNumOfInsertions;
		// Predicted clusters in each reservoir
		std::vector<PredictionType> mPredictions;
		// Number of trees to use
		unsigned mNumOfTrees;
		// Num of reservoirs
		unsigned mNumOfReservoirs;
		// Maximal samples in each reservoir
		unsigned mMaxNumOfSamplesPerReservoir;
		// Gaussion values of each sample in a reservoir
		std::vector<float> mGaussianValues;

		// Clustering parameters
		float mSigma;
		float mTau;
		// Maximal clusters in each reservoir
		unsigned mMaxNumOfClustersPerReservoir;
		// Minimal samples in each cluster
		unsigned mMinNumOfSamplesPerCluster;
		// Number of clusters in a reservoir
		std::vector<unsigned> mNumOfClustersPerReservoir;
		std::vector<unsigned> mPreviousIndicies;
		// Cluster center indices in each reservoir
		std::vector<unsigned> mClusterIndicies;
		// Number of samples in one cluster and cluster ID 
		std::vector<ClusterIDSize> mClusterIDSize;
		// Selected cluster ID
		std::vector<unsigned> mSelectedClusterIDs;

		// Random number generator
		RandomGenerator mRandomGenerator;

		//**************************************PRIVATE FUNCTIONS**************************************//
	private:
		// Initialize variables
		void Initialize();
		// Insert a sample to the reservoirs
		void InsertSample(unsigned selectedReservoirIndex, const SampleType& sample);
		// Compute gaussian value of samples within the range of 3*mSigma in a reservoir 
		void computeGaussianValues(unsigned reservoirIndex);
		// Compute cluster centers of samples in a reservoir
		void computeClusterCenters(unsigned reservoirIndex);
		// Compute clusters from calculated cluster centers
		void computeClusters(unsigned reservoirIndex);
		// Select clusters that are not larger than mMaxNumOfClustersPerReservoir
		void selectClusters(unsigned reservoirIndex);
		// Compute statistic in each cluster
		void computeStatistics(unsigned reservoirIndex);
		// Predict the modes of a sample
		void extractMode(const std::vector<unsigned>& reservoirIndices, unsigned sampleIndex, std::vector<PredictionType>& predictions);
	};

	template<typename SampleType, typename PredictionType>
	Reservoirs<SampleType, PredictionType>::Reservoirs(unsigned numOfTrees, unsigned numOfReservoirs, unsigned maxNumOfSamplesPerReservoir, float sigma,
		float tau, unsigned maxNumOfClustersPerReservoir, unsigned minNumOfSamplesPerCluster)
		:mNumOfTrees(numOfTrees), numOfReservoirs(numOfReservoirs), mMaxNumOfSamplesPerReservoir(maxNumOfSamplesPerReservoir), mSigma(sigma),
		tau(mTau), mMaxNumOfClustersPerReservoir(maxNumOfClustersPerReservoir), mMinNumOfSamplesPerCluster(minNumOfSamplesPerCluster)
	{
		// Initialize
		Initialize();
		// Set random number generator
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		mRandomGenerator.resize(numOfInitialHypothesis);
		mRandomGenerator.setSeed(seed);
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::InsertSamples(const std::vector<SampleType>& samples, const std::vector<unsigned>& reservoirIndices)
	{
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (size_t sampleIndex = 0, sampleIndexEnd = samples.size(); sampleIndex < sampleIndexEnd; sampleIndex++)
		{
			const SampleType& sample = samples[sampleIndex];
			for (unsigned treeIndex = 0; treeIndex < mNumOfTrees; treeIndex++)
			{
				unsigned sampleStartIndex = sampleIndex*mNumOfTrees + treeindex;
				unsigned reservoirIndex = reservoirIndices[sampleStartIndex];
				// Insert a sample to the reservoir
				InsertSample(reservoirIndex, sample);
			}
		}
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::findClusters(unsigned updateStartIndex, unsigned numOfUpdatedLeaves)
	{
		unsigned reservoirEndIndex = updateStartIndex + numOfUpdatedLeaves;
		unsigned tempReservoirCount = std::min(reservoirEndIndex, mNumOfReservoirs);
		// Cluster samples in each reservoir
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (unsigned reservoirIndex = updateStartIndex; reservoirIndex < tempReservoirCount; reservoirIndex++)
		{
			computeGaussianValues(reservoirIndex);

			computeClusterCenters(reservoirIndex);

			computeClusters(reservoirIndex);

			selectClusters(reservoirIndex);

			computeStatistics(reservoirIndex);
		}
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::extractModes(const std::vector<unsigned>& reservoirIndices, std::vector<PredictionType>& predictions)
	{
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (unsigned sampleIndex = 0; sampleIndex < numOfSamples; sampleIndex++)
		{
			extractMode(reservoirIndicies, sampleIndex, predictions);
		}
	}

	template<typename SampleType, typename PredictionType>
	inline bool Reservoirs<SampleType, PredictionType>::comparision(const ClusterIDSize & c1, const ClusterIDSize & c2)
	{
		return c1.mSize > c2.mSize;
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::reset()
	{
	}


	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::Initialize()
	{
		unsigned maxNumOfSamples = mNumOfReservoirs*mMaxNumOfSamplesPerReservoir;

		mReservoirs.resize(maxNumOfSamples);
		mNumOfSamplesInReservoirs.resize(mNumOfReservoirs);
		mNumOfInsertions.resize(mNumOfReservoirs);
		mPredictions.resize(mNumOfReservoirs);

		mGaussianValues.resize(maxNumOfSamples);
		mNumOfClustersPerReservoir.resize(mNumOfReservoirs);
		mPreviousIndicies.resize(maxNumOfSamples);
		mClusterIndicies.resize(maxNumOfSamples);
		mClusterIDSize.resize(maxNumOfSamples);
		mSelectedClusterIDs.resize(mMaxNumOfClustersPerReservoir*mNumOfReservoirs);
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::InsertSample(unsigned selectedReservoirIndex, const SampleType & sample)
	{
		unsigned reservoirStartIndex = selectedReservoirIndex*mMaxNumOfSamplesPerReservoir;
		unsigned reservoirSize = 0;
#ifdef OPENMP
#pragma omp atomic capture
#endif // OPENMP
		numOfInsertions = mNumOfInsertions[selectedReservoirIndex]++;

		if (numOfInsertions < mMaxNumOfSamplesPerReservoir)
		{
			mReservoirs[reservoirStartIndex + reservoirSize - 1] = sample;
#ifdef OPENMP
#pragma omp atomic capture
#endif // OPENMP
			++mNumOfSamplesInReservoirs[selectedReservoirIndex];
		}
		else
		{
			int randomIndex = mRandomGenerator.uniformIntDistribution(0, reservoirSize - 1);

			if (randomIndex < mMaxNumOfSamplesPerReservoir)
				mReservoirs[reservoirStartIndex + randomIndex] = sample;
		}
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::computeGaussianValues(unsigned reservoirIndex)
	{
		unsigned sampleSize = mNumOfSamplesInReservoirs[reservoirIndex];
		unsigned reservoirStartIndex = reservoirIndex*mMaxNumOfSamplesPerReservoir;
		float range = 9.0f*mSigma*mSigma;
		float coef = -0.5f / (mSigma*mSigma);

		for (unsigned sampleIndex = 0; sampleIndex < sampleSize; sampleIndex++)
		{
			float gaussianValue = 0.0f;
			unsigned currSampleIndex = reservoirStartIndex + sampleIndex;
			const SampleType& currSample = mReservoirs[currSampleIndex];

			for (unsigned innerSampleInd = 0; innerSampleInd < sampleSize; innerSampleInd++)
			{
				const SampleType& testSample = mReservoirs[reservoirStartIndex + innerSampleIndex];
				float squaredDistance = computeSquaredDistance(currSample, testSample);

				if (squaredDistance < range)
					gaussianValue += expf(coef*squaredDistance);
			}

			mGaussianValues[currSampleIndex] = gaussianValue;
		}
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::computeClusterCenters(unsigned reservoirIndex)
	{
		unsigned sampleSize = mNumOfSamplesInReservoirs[reservoirIndex];
		unsigned reservoirStartIndex = reservoirIndex*mMaxNumOfSamplesPerReservoir;
		float minDistance = mTau*mTau;

		for (unsigned sampleIndex = 0; sampleIndex < sampleSize; sampleIndex++)
		{
			unsigned currSampleIndex = reservoirStartIndex + sampleIndex;
			const SampleType& currSample = mReservoirs[currSampleIndex];
			float currGaussianValue = mGaussianValues[currSampleIndex];
			unsigned selectedIndex = sampleIndex;
			unsigned centerIndex = UINT_MAX;

			for (unsigned innerSampleInd = 0; innerSampleInd < sampleSize; innerSampleInd++)
			{
				if (innerSampleInd == sampleIndex) continue;

				const SampleType& testSample = mReservoirs[reservoirStartIndex + innerSampleIndex];
				float squaredDistance = computeSquaredDistance(currSample, testSample);

				if (squaredDistance < minDistance)
				{
					minDistance = squaredDistance;
					selectedIndex = innerSampleInd;
				}
			}

			if (selectedIndex = sampleIndex)
			{
#ifdef OPENMP
#pragma omp atomic capture
#endif // OPENMP
				centerIndex = mNumOfClustersPerReservoir[reservoirIndex]++;
			}
			mPreviousIndicies[currSampleIndex] = selectedIndex;
			mClusterIndicies[currSampleIndex] = centerIndex;
		}
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::computeClusters(unsigned reservoirIndex)
	{
		unsigned sampleSize = mNumOfSamplesInReservoirs[reservoirIndex];
		unsigned reservoirStartIndex = reservoirIndex*mMaxNumOfSamplesPerReservoir;

		for (unsigned sampleIndex = 0; sampleIndex < sampleSize; sampleIndex++)
		{
			unsigned currSampleIndex = reservoirStartIndex + sampleIndex;

			unsigned currIndex = sampleIndex;
			unsigned selectedIndex = mPreviousIndicies[currSampleIndex];

			while (selectedIndex != currIndex)
			{
				currIndex = selectedIndex;
				selectedIndex = mPreviousIndicies[reservoirStartIndex + selectedIndex];
			}

			unsigned clusterIndex = mClusterIndicies[reservoirStartIndex + selectedIndex];
			mClusterIndicies[currSampleIndex] = clusterIndex;

			if ((clusterIndex < mMaxNumOfSamplesPerReservoir)
			{
#ifdef OPENMP
#pragma omp atomic capture
#endif // OPENMP
				mClusterIDSize[reservoirStartIndex + clusterIndex].mSize++;
			}
		}
	}

	template<typename SampleType, typename PredictionType>
	void Reservoirs<SampleType, PredictionType>::selectClusters(unsigned reservoirIndex)
	{
		unsigned numOfClusters = mNumOfClustersPerReservoir[reservoirIndex];
		unsigned reservoirStartIndex = reservoirIndex*mMaxNumOfSamplesPerReservoir;
		unsigned reservoirEndIndex = reservoirStartIndex + numOfClusters;
		std::sort(mClusterIDSize.begin() + reservoirStartIndex, mClusterIDSize.begin() + reservoirEndIndex, Reservoirs::comparision);

		unsigned clusterSize = min(mMaxNumOfClustersPerReservoir, numOfClusters);
		for (unsigned cIndex = 0; cIndex < clusterSize; cIndex++)
		{
			const ClusterIDSize & currClusterIDSize = mClusterIDSize[reservoirStartIndex + cIndex];
			if (currClusterIDSize.mSize < mMinNumOfSamplesPerCluster)
				break;

			mSelectedClusterIDs[reservoirIndex*mMaxNumOfClustersPerReservoir + cIndex] = currClusterIDSize.mClusterId;
		}
	}

	template<typename SampleType, typename PredictionType>
	void CamRelocalizer::Reservoirs<SampleType, PredictionType>::computeStatistics(unsigned reservoirIndex)
	{
		unsigned sampleSize = mNumOfSamplesInReservoirs[reservoirIndex];
		unsigned reservoirStartIndex = reservoirIndex*mMaxNumOfSamplesPerReservoir;
		PredictionType& currPredictionType = mPredictions[reservoirIndex];

#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (unsigned clusterIndex = 0; clusterIndex < mMaxNumOfClustersPerReservoir; clusterIndex++)
		{
			unsigned currClusterID = mSelectedClusterIDs[reservoirStartIndex*mMaxNumOfClustersPerReservoir + clusterIndex];
			if (currClusterID > mMaxNumOfClustersPerReservoir) break;

			unsigned currModeIndex = UINT_MAX;
#ifdef OPENMP
#pragma omp atomic capture
#endif // OPENMP
			currModeIndex = currPredictionType.mSize++;

			auto& currMode = currPredictionType.mPredictedModes[currModeIndex];
			Eigen::Vector3i colorMean(0);
			Eigen::Vector3f mu(0.0f);
			Eigen::Matrix3f sigma = Eigen::Matrix3f::Zero();
			float coef = 0.0f;
			unsigned numOfSamplesInCluster = 0;

			// Compute mean values of current cluster
			for (unsigned sampleIndex = 0; sampleIndex < sampleSize; sampleIndex++)
			{
				unsigned sampleClusterID = mClusterIndicies[reservoirStartIndex + sampleIndex];
				if (sampleClusterID == currClusterID)
				{
					const SampleType& sample = mReservoirs[reservoirStartIndex + sampleIndex];
					colorMean += sample.mColor;
					mu += sample.mPosition;
					numOfSamplesInCluster++;
				}
			}

			mu /= numOfSamplesInCluster;
			colorMean /= static_cast<float>(numOfSamplesInCluster);
			coef = static_cast<float>(numOfSamplesInCluster) / static_cast<float>(sampleSize);

			for (unsigned sampleIndex = 0; sampleIndex < sampleSize; sampleIndex++)
			{
				unsigned sampleClusterID = mClusterIndicies[reservoirStartIndex + sampleIndex];
				if (sampleClusterID == currClusterID)
				{
					const SampleType& sample = mReservoirs[reservoirStartIndex + sampleIndex];
					colorMean += sample.mColor;
					mu += sample.mPosition;
					numOfSamplesInCluster++;

					for (unsigned i = 0; i < 3; i++)
					{
						for (unsigned j = 0; j < 3; j++)
						{
							sigma(i, j) += (sample.mPosition(i) - mu(i))*(sample.mPosition(j) - mu(j));
						}
					}
				}
			}

			currMode.mNumOfSamples = numOfSamplesInCluster;
			currMode.mCoef = coef;
			currMode.mDeterminant = sigma.determinant();
			currMode.mMu = mu;
			currMode.mColorMean = colorMean;
			currMode.mSigmaInv = sigma.inverse();
		}
	}

	template<typename SampleType, typename PredictionType>
	void CamRelocalizer::Reservoirs<SampleType, PredictionType>::extractMode(const std::vector<unsigned>& reservoirIndices, unsigned sampleIndex, std::vector<PredictionType>& predictions)
	{
		unsigned sampleStarIndex = sampleIndex*mNumOfTrees;
		std::vector<PredictionType> tempPredictions(mNumOfTrees);
		for (unsigned treeIndex = 0; treeIndex < mNumOfTrees; treeIndex++)
		{
			tempPredictions[treeIndex] = mPredictions[reservoirIndices[sampleStarIndex + treeIndex]];
		}

		// Merge predictions from all trees and sort them in descending order
		std::vector<unsigned> tempModeIndices(mNumOfTrees, 0);
		PredictionType currPrediction;
		unsigned maxClusterSize = mMaxNumOfClustersPerReservoir*mNumOfTrees;

		while (currPrediction.mSize < maxClusterSize)
		{
			unsigned maxTreeIndex = 0;
			unsigned maxTreeNumOfSamples = 0;

			for (unsigned treeIndex = 0; treeIndex < mNumOfTrees; treeIndex)
			{
				unsigned modeIndex = tempModeIndices[treeIndex];
				if (modeIndex >= tempPredictions[treeIndex].mSize)
					continue;
				const auto& predictedMode = tempPredictions[treeIndex].mPredictedModes[modeIndex];

				if (predictedMode.mNumOfSamples > maxTreeNumOfSamples)
				{
					maxTreeNumOfSamples = predictedMode.mNumOfSamples;
					maxTreeIndex = treeIndex;
				}
			}

			if (maxTreeNumOfSamples == 0)
				break;
			unsigned maxModeIndex = tempModeIndices[maxTreeIndex]++;
			unsigned position = currPrediction.mSize++;
			currPrediction.mPredictedModes[position] = tempPredictions[maxTreeIndex].mPredictedModes[maxModeIndex];
		}

		predictions[sampleIndex] = currPrediction;
	}
}