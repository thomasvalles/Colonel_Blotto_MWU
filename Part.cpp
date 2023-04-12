#include "Part.h"

Eigen::ArrayXi rand_k_el(Eigen::ArrayXi elements, int k) {
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, 1.0);
	//std::cout << "here" << '\n';
	Eigen::ArrayXi a = Eigen::ArrayXi::Zero(k);
	
	int c1 = k;
	int c2 = elements.rows();
	int i = 0;
	int x = 0;
	while (c1 > 0) {
		if (dis(gen) <= (double)(c1) / c2) {

			a(x) = (elements(i));
			++x;
			c1--;
		}
		c2--;
		i++;
	}
	return a;
}

Eigen::ArrayXi rand_comp_n_k(int n, int k) {
	auto v = Eigen::ArrayXi::LinSpaced(n + k, 1, n + k);

	auto a = rand_k_el(v, k - 1);
	

	Eigen::ArrayXi r = Eigen::ArrayXi::Zero(k);
	r(0) = a(0) - 1;
	for (size_t j = 1; j < k - 1; ++j) {
		r(j) = a(j) - a(j - 1) - 1;
	}
	r(k - 1) = n + k - 1 - a(k - 2);
	//std::cout << "r: " << '\n' <<  r << '\n';
	return r;
}