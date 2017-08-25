#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace CamRelocalizer
{
	template<typename DescriptorType, typename NodeType>
	class ForestPredictor
	{
	public:
		ForestPredictor(const std::string& forestFile);
		

		//***********************************PUBLIC FUNCTIONS***********************************//
		// Predict the leaf indices of the descriptors
		void predict(const std::vector<DescriptorType>& descriptors, std::vector<unsigned>& leafIndices);
		// Return number of trees
		unsigned getNumOfTrees();

		void reset();
	private:
		//***********************************PRIVATE VARIABLES***********************************//
		// Nodes in the forest
		std::vector<NodeType> mNodes;
		// Number of trees
		unsigned mNumOfTrees;
		// Number of nodes in each tree
		std::vector<unsigned> mNumOfNodesPerTree;
		// Number of leaves in each tree
		std::vector<unsigned> mNumOfLeavesPerTree;

		//***********************************PRIVATE FUNCTIONS***********************************//
		// Load trained forest structure
		void loadForest(const std::string& filename);
		// Predict leaf indices of a descriptor
		void predictDescriptor(unsigned descriptorIndex, std::vector<DescriptorType>& descriptor, std::vector<unsigned>& leafIndices);
	};

	template<typename DescriptorType, typename NodeType>
	ForestPredictor<DescriptorType, typename NodeType>::ForestPredictor(const std::string & forestFile)
	{
		loadForest(forestFile);
	}

	template<typename DescriptorType, typename NodeType>
	void ForestPredictor<DescriptorType, typename NodeType>::predict(const std::vector<DescriptorType>& descriptors, std::vector<unsigned>& leafIndices)
	{
#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
		for (size_t dIndex = 0, dIndexEnd=descriptors.size(); dIndex < dIndexEnd; dIndex++)
		{
			predictDescriptor(dIndex, descriptors, leafIndices);
		}
	}

	template<typename DescriptorType, typename NodeType>
	unsigned ForestPredictor<DescriptorType, NodeType>::getNumOfTrees()
	{
		return mNumOfTrees;
	}

	template<typename DescriptorType, typename NodeType>
	void ForestPredictor<DescriptorType, typename NodeType>::reset()
	{

	}

	template<typename DescriptorType, typename NodeType>
	void ForestPredictor<DescriptorType, typename NodeType>::loadForest(const std::string & filename)
	{
		std::ifstream input(filename);

		if (input.is_open())
		{
			input >> mNumOfTrees;

			unsigned maxNumOfNodesPerTree = 0;
			for (unsigned treeIndex = 0; treeIndex < mNumOfTrees; treeIndex++)
			{
				unsigned numOfNodesPerTree, numOfLeavesPerTree;
				input >> numOfNodesPerTree >> numOfLeavesPerTree;

				mNumOfNodesPerTree.push_back(numOfNodesPerTree);
				mNumOfLeavesPerTree.push_back(numOfLeavesPerTree);

				maxNumOfNodesPerTree = std::max(maxNumOfNodesPerTree, numOfNodesPerTree);
			}

			mNodes.resize(maxNumOfNodesPerTree*mNumOfTrees);
			for (unsigned treeIndex = 0; treeIndex < mNumOfTrees; treeIndex++)
			{
				for (unsigned nodeIndex = 0; nodeIndex < mNumOfNodesPerTree[treeIndex]; nodeIndex++)
				{
					NodeType& currNode = mNodes[nodeIndex*mNumOfTrees + treeIndex];

					input >> currNode.mLeftChildIndex >> currNode.mLeafIndex >> currNode.mSelectedFeatureIndex >> currNode.mThreshold;
				}
			}
		}
	}
	
	template<typename DescriptorType, typename NodeType>
	void ForestPredictor<DescriptorType, NodeType>::predictDescriptor(unsigned descriptorIndex, std::vector<DescriptorType>& descriptor, std::vector<unsigned>& leafIndices)
	{
		const DescriptorType& descriptor = descriptors[descriptorIndex];
		unsigned leafStartIndex = descriptorIndex*mNumOfTrees;

		for (unsigned treeIndex = 0; treeIndex < mNumOfTrees; treeIndex++)
		{
			unsigned currNodeIndex = 0;
			bool isLeaf = false;
			
			while (true)
			{
				const currNode& = mNodes[currNodeIndex*mNumOfTrees + treeIndex];
				isLeaf = currNode.mLeftChildIndex == -1;
				
				if (isLeaf) break;

				currNodeIndex = descriptor.mData[currNode.mSelectedFeatureIndex] > currNode.mThreshold ? currNode.mLeftChildIndex : currNode.mLeftChildIndex + 1;
			}

			leafIndices[leafStartIndex + treeIndex] = mNodes[currNodeIndex*mNumOfTrees + treeIndex].mLeafIndex + (treeIndex > 0 ? mNumOfLeavesPerTree[treeIndex - 1] : 0));
		}
	}
}