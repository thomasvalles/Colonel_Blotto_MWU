#pragma once
#include<vector>
#include<algorithm>
#include<cmath>
#include<numeric>
#include<iostream>
#include<random>
#include<Eigen>

/**
These functions are used to compute a random partition of an integer into k parts,
used only in the first round of play. Translated the python code found at:
https://pythonhosted.org/combalg-py/
*/

/**
Gets a random k element subset of an input vector
@param elements: The vector to get the subset from
@param k: Size of subset
@return A random k element subset of the input vector
*/
Eigen::ArrayXi rand_k_el(Eigen::ArrayXi elements, int k);

/**
Gets a random partition of an integer n into k parts. May have zeros.
@param n: Number to partition
@param k: Number of parts
@return: Vector of length k representing the partition. Sum of elements is n.
*/
Eigen::ArrayXi rand_comp_n_k(int n, int k);

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
double incbeta(double a, double b, double x);