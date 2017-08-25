#pragma once
#include<random>
namespace CamRelocalizer
{
	class RandomGenerator
	{
	public:
		RandomGenerator(){};

		RandomGenerator(unsigned seed) :mGenerator(seed) {};
		~RandomGenerator()
		{};

		// Generate random int number of the range [a,b]
		inline int uniformIntDistribution(int a, int b)
		{
			std::uniform_int_distribution<int> dist(a, b);
			return dist(mGenerator);
		};

		inline void setSeed(unsigned seed)
		{
			mGenerator.seed(seed);
		};
	private:
		std::mt19937 mGenerator;
	};
}