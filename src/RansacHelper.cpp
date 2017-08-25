#include"RansacHelper.h"
//#define DEBUG_INFO

namespace CamRelocalizer
{
	CostFunctor::CostFunctor(const Eigen::Vector3d& xws, const Eigen::Matrix3d& covarianceInv, const Eigen::Vector3d& xcs)
		:mXws(xws), mCovarianceInv(covarianceInv), mXcs(xcs) 
	{}

	int sampleOnePoints(RandomGenerator& randomGen, std::vector<int>& selectedPointIndices, int maxNum)
	{
		int randInt = -1;
		for (int iter = 0; randInt < 0 && iter < 40; iter++)
		{
			int currRandInt = randomGen.uniformIntDistribution(0, maxNum - 1);
			int isSelected = 1;
#ifdef OPENMP
#pragma omp automic capture
#endif // OPENMP
			isSelected = selectedPointIndices[currRandInt]++;

			if (isSelected == 0) randInt = currRandInt;
		}

		return randInt;
	}

	Eigen::Matrix4f kabschAlgorithm(const Eigen::Vector3f xcs[3], const Eigen::Vector3f xws[3])
	{
		Eigen::Matrix4f H;

		// Construct Mat
		Eigen::Matrix3f P;
		Eigen::Matrix3f Q;
		P.row(0) = xcs[0];
		P.row(1) = xcs[1];
		P.row(2) = xcs[2];

		Q.row(0) = xws[0];
		Q.row(1) = xws[1];
		Q.row(2) = xws[2];

#ifdef DEBUG_INFO
		std::cerr << "P: " << P << std::endl;
		std::cerr << "Q: " << Q << std::endl;
#endif

		// Centroids of two point pairs
		Eigen::Vector3f centroidP = P.rowwise().mean();
		Eigen::Vector3f centroidQ = Q.rowwise().mean();

#ifdef DEBUG_INFO
		std::cerr << "CentroidP: " << centroidP << std::endl;
		std::cerr << "CentroidQ: " << centroidQ << std::endl;
#endif

		// Center the points
		Eigen::Matrix3f PP = P.colwise() - centroidP;
		Eigen::Matrix3f QQ = Q.colwise() - centroidQ;

#ifdef DEBUG_INFO
		std::cerr << "PP: " << PP << std::endl;
		std::cerr << "QQ: " << QQ << std::endl;
#endif

		// Covariance matrix

		Eigen::Matrix3f A = PP*QQ.transpose();
		Eigen::Matrix3f U, S, V;
		Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
		U = svd.matrixU();
		S = svd.singularValues().asDiagonal();
		V = svd.matrixV();

#ifdef DEBUG_INFO
		std::cerr << "SVD Of " << A << std::endl;
		std::cerr << "U: " << U << std::endl;
		std::cerr << "S: " << S << std::endl;
		std::cerr << "V: " << V << std::endl;
#endif

		// Check if rotation matrix needs correction
		Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
		if ((U*V.transpose()).determinant() < 0)
		{
			I(2, 2) = -1;
		}
		Eigen::Matrix3f R = V*I*U.transpose();


		// Translation matrix
		Eigen::Vector3f t = -R*centroidP + centroidQ;

		H = Eigen::Matrix4f::Identity();
		H.block<3, 3>(0, 0) = R;
		H.block<3, 1>(0, 3) = t;

		return H;
	}
}