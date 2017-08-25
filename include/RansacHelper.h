#pragma once
#include<Eigen\Dense>
#include<ceres\ceres.h>
#include<ceres\rotation.h>
#include"RandomGenerator.h"
//***********************DEFINITIONS***********************//
#define M_PI 3.14159265358979323846

namespace CamRelocalizer
{
	//*****************************TYPEDEFINES**************************//
	typedef Eigen::Matrix<double, 6, 1> Vector6d;

	//*****************************STRUCT**************************//
	struct CostFunctor
	{
		CostFunctor(const Eigen::Vector3d& xws, const Eigen::Matrix3d& covarianceInv, const Eigen::Vector3d& xcs);

		template<typename T>
		bool operator () (const T* const R_t, T* residual) const
		{
			T transformedXws[3];
			const T xcs[3] = { T(mXcs(0)), T(mXcs(1)), T(mXcs(2)) };
			const T xws[3] = { T(mXws(0)), T(mXws(1)), T(mXws(2)) };

			ceres::AngleAxisRotatePoint(R_t, xcs, transformedXws);
			
			transformedXws[0] += R_t[3];
			transformedXws[1] += R_t[4];
			transformedXws[2] += R_t[5];
			
			Eigen::Matrix<T, 3, 1> diff(transformedXws[0] - xws[0], transformedXws[1] - xws[1], transformedXws[2] - xws[2]);
			Eigen::Matrix<T, 3, 3> covarianceInv;
			covarianceInv << T(mCovarianceInv(0, 0)), T(mCovarianceInv(0, 1)), T(mCovarianceInv(0, 2)),
				T(mCovarianceInv(1, 0)), T(mCovarianceInv(1, 1)), T(mCovarianceInv(1, 2)),
				T(mCovarianceInv(2, 0)), T(mCovarianceInv(2, 1)), T(mCovarianceInv(2, 2));

			residual[0] = diff.transpose()*covarianceInv*diff;
			return true;
		};

		const Eigen::Vector3d mXws;
		const Eigen::Matrix3d mCovarianceInv;
		const Eigen::Vector3d mXcs;
	};

	int sampleOnePoints(RandomGenerator& randomGen, std::vector<int>& selectedPointIndices, int maxNum);

	template<typename HypothesisType, typename PointType, typename PredictionType>
	float computeEnergy(HypothesisType& currHypothesis, const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData, const std::vector<int>& pointIndices, float inlierThreshold, bool isSelectInliers)
	{
		size_t numOfPoints = pointIndices.size();
		// Check if select inliers
		if (isSelectInliers)
		{
			currHypothesis.mInlierIndices.resize(numOfPoints);
			currHypothesis.mBestModeIndices.resize(numOfPoints);
		}

		float totalEnergy = 0;

		for (size_t pointIndex = 0; pointIndex < numOfPoints; pointIndex++)
		{
			int currPointIndex = pointIndices[pointIndex];
			const auto& predictedModes = predictedData[currPointIndex].mPredictedModes;
			const Eigen::Vector3f& cameraSpaceCoordinate = points[currPointIndex].mPosition;
			Eigen::Vector3f transformedWorldCoordinate = currHypothesis.mPose.block<3,3>(0,0)*cameraSpaceCoordinate+ currHypothesis.mPose.block<3, 1>(0, 3);

			float maxEnergy = 0;
			size_t modeSize = predictedModes.size();
			float ratio = powf(2.0f*M_PI, 1.5);
			int bestModeIndex = -1;

			for (size_t modeIndex = 0; modeIndex < modeSize; modeIndex++)
			{
				// Compute multivariante normal distribution
				const auto& currMode = predictedModes[modeIndex];
				const Eigen::Vector3f diffV = transformedWorldCoordinate - currMode.mMu;
				const float squaredDistance = diffV.transpose()*currMode.mSigmaInv*diffV;

				const float normalizationRatio = 1.0f / (ratio*sqrtf(currMode.mDeterminant));
				const float exponentialValue = expf(-0.5f*squaredDistance);

				const float currEnergy = currMode.mCoef*normalizationRatio*exponentialValue;

				if (currEnergy > maxEnergy)
				{
					maxEnergy = currEnergy;
					bestModeIndex = modeIndex;
				}
			}

			if (bestModeIndex < 0)
			{
				bestModeIndex = -1;

				for (size_t modeIndex = 0; modeIndex < modeSize; modeIndex++)
				{
					// Compute multivariante normal distribution
					const auto& currMode = predictedModes[modeIndex];
					const Eigen::Vector3f diffV = transformedWorldCoordinate - currMode.mMu;
					const float squaredDistance = diffV.transpose()*currMode.mSigmaInv*diffV;

					const float normalizationRatio = 1.0f / (ratio*sqrtf(currMode.mDeterminant));
					const float exponentialValue = expf(-0.5f*squaredDistance);

					const float currEnergy = currMode.mCoef*normalizationRatio*exponentialValue;

					if (currEnergy > maxEnergy)
					{
						maxEnergy = currEnergy;
						bestModeIndex = modeIndex;
					}
				}
			}

			if (isSelectInliers&&bestModeIndex>=0)
			{
				const auto& currMode = predictedModes[bestModeIndex];
				if ((currMode.mMu-transformedWorldCoordinate).norm()>=inlierThreshold)
				{
					currHypothesis.mInlierIndices[pointIndex] = -1;
				}
				else
				{
					currHypothesis.mInlierIndices[pointIndex] = currPointIndex;
					currHypothesis.mBestModeIndices[pointIndex] = bestModeIndex;
				}
			}

			// Compute minus logarithm
			if (maxEnergy < 1e-6) maxEnergy = 1e-6;
			maxEnergy = -log10f(maxEnergy);
			totalEnergy += maxEnergy;
		}

		return (totalEnergy / static_cast<float>(numOfPoints));
	}

	Eigen::Matrix4f kabschAlgorithm(const Eigen::Vector3f xcs[3], const Eigen::Vector3f xws[3]);

	template<typename PointType, typename PredictionType, typename HypothesisType>
	bool generateOneHypothesis(const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData, HypothesisType& hypothesis,int maxNumOfIters, bool isRandomMode, int indensityThreshold, float minPointDistance, float minTranslationError, RandomGenerator& randGen)
	{
		int selectedPointIndex = 0;
		int numOfPoints = static_cast<int>(points.size());
		Eigen::Vector3f selectedWorldPositions[3];
		Eigen::Vector3f selectedCameraPositions[3];
		
		// Sampling points until reaching max iteration or picking enough points
		for (int iterIndex = 0; selectedPointIndex<3 && iterIndex<maxNumOfIters; iterIndex++)
		{
			// Randomly choose a points from points
			int randPointIndex = randGen.uniformIntDistribution(0, numOfPoints-1);
			const PointType& selectedPoint = points[randPointIndex];

			// A.1 Check rgb indensity distance
			const auto& predictedModes = predictedData[randPointIndex].mPredictedModes;
			int randModeIndex = isRandomMode ? randGen.uniformIntDistribution(0, static_cast<int>(predictedModes.size()-1)) : 1;
			const auto& selectedMode = predictedModes[randModeIndex];

			if (selectedPointIndex == 0)
			{
				Eigen::Vector3i colorDiff = selectedPoint.mColor - selectedMode.mColorMean;
				if (abs(colorDiff[0]) > indensityThreshold || abs(colorDiff[1]) > indensityThreshold || abs(colorDiff[2]) > indensityThreshold)
					continue;
			}

			// A.2 Check if 3 points are far from each other
			bool isFar = true;
			for (int pointIndex = 0; isFar && pointIndex < selectedPointIndex; pointIndex++)
			{
				Eigen::Vector3f worldDiff = selectedMode.mMu - selectedWorldPositions[pointIndex];
				if (worldDiff.norm() < minPointDistance)
					isFar = false;
			}

			if (!isFar) continue;

			// A.3 Check if the transfrom is rigid enough
			bool isRigidTransformation = true;
			for (int pointIndex = 0; isRigidTransformation && pointIndex < selectedPointIndex; pointIndex++)
			{
				Eigen::Vector3f cameraDiff = selectedPoint.mPosition - selectedCameraPositions[pointIndex];
				Eigen::Vector3f worldDiff = selectedMode.mMu - selectedWorldPositions[pointIndex];
				if (fabsf(cameraDiff.norm() - worldDiff.norm()) > minTranslationError)
					isRigidTransformation = false;
			}

			if (!isRigidTransformation) continue;

			selectedCameraPositions[selectedPointIndex]= selectedPoint.mPosition;
			selectedWorldPositions[selectedPointIndex]= selectedMode.mMu;
			selectedPointIndex++;
		}

		// Check if succeed or not
		if (selectedPointIndex == 3)
		{
			hypothesis.setPointPairs(selectedCameraPositions, selectedWorldPositions);
			return true;
		}
		else
			return false;
	}
	
	template<typename HypothesisType, typename PointType, typename PredictionType>
	void optimizationLM(HypothesisType& hypothesis, const std::vector<PointType>& points, const std::vector<PredictionType>& predictedData)
	{
		// Transform the pose to Lie algebra
		Vector6d x;
		Eigen::Matrix3d currPose = hypothesis.mPose.block<3, 3>(0, 0).cast<double>();
		ceres::RotationMatrixToAngleAxis(&currPose(0, 0), &x(0));
		x.tail<3>() = hypothesis.mPose.block<3, 1>(0, 3).cast<double>();

		ceres::Problem problem;
		for (size_t ptIndex = 0, ptEnd= hypothesis.mInlierIndices.size(); ptIndex < ptEnd; ptIndex++)
		{
			int pointIndex = hypothesis.mInlierIndices[ptIndex];
			if (pointIndex < 0) continue;

			int bestModeIndex = hypothesis.mBestModeIndices[ptIndex];
			const auto& predictedModes = predictedData[pointIndex].mPredictedModes;
			const Eigen::Vector3d xws = predictedModes[bestModeIndex].mMu.cast<double>();
			const Eigen::Matrix3d covarianceInv = predictedModes[bestModeIndex].mSigmaInv.cast<double>();
			const Eigen::Vector3d xcs = points[pointIndex].mPosition.cast<double>();
			//ceres::LossFunction* lossFunc = new ceres::HuberLoss(1.0);

			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<CostFunctor, 1, 6>(
				new CostFunctor(xws, covarianceInv, xcs));

			problem.AddResidualBlock(costFunction, NULL, &(x(0)));
		}

		ceres::Solver::Options options;

		options.minimizer_type = ceres::TRUST_REGION;
		//options.linear_solver_type = ceres::DENSE_SCHUR;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.preconditioner_type = ceres::JACOBI;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		//options.dogleg_type = ceres::TRADITIONAL_DOGLEG;
		options.use_nonmonotonic_steps = false;
		options.use_inner_iterations = false;
		options.max_num_iterations = 100;
		options.minimizer_progress_to_stdout = true;
		//options.eta = 1e-5;
		options.function_tolerance = std::numeric_limits<double>::epsilon();
		options.gradient_tolerance = std::numeric_limits<double>::epsilon();
		options.parameter_tolerance = std::numeric_limits<double>::epsilon();

		options.inner_iteration_ordering.reset(new ceres::ParameterBlockOrdering);
		options.inner_iteration_ordering->AddElementToGroup(&(x(0)), 0);

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		std::cerr << summary.FullReport() << "\n";

		ceres::AngleAxisToRotationMatrix(&x(0), &currPose(0, 0));
		hypothesis.mPose.block<3, 3>(0, 0) = currPose.cast<float>();
		hypothesis.mPose.block<3, 1>(0, 3) = x.tail<3>().cast<float>();
	}
}