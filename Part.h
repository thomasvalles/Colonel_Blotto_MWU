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