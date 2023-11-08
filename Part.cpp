#include "Part.h"

Eigen::ArrayXi rand_k_el(Eigen::ArrayXi elements, int k) {
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, 1.0);
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
	auto v = Eigen::ArrayXi::LinSpaced(n + k - 1, 1, n + k - 1);
	auto a = rand_k_el(v, k - 1);

	Eigen::ArrayXi r = Eigen::ArrayXi::Zero(k);
	r(0) = a(0) - 1;
	for (size_t j = 1; j < k - 1; ++j) {
		r(j) = a(j) - a(j - 1) - 1;
	}
	r(k - 1) = n + k - 1 - a(k - 2);

	return r;
}

/*
 * zlib License
 *
 * Regularized Incomplete Beta Function
 *
 * Copyright (c) 2016, 2017 Lewis Van Winkle
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */


