#include "CB.h"
#include "Part.h"
#include<iostream>
#include<fstream>
#include<string>
#include<chrono>
#include<thread>
#include<random>
#include<cmath>
#include<filesystem>


/**
Runs a specified test. Outputs test results to file named by parameters
@param battles: the number of battles to run
@param soldiers: vector of soldiers for the two players
@param optimistic: true if optimistic hedge
@param random: true of weights are to be randomly selected
*/
void run_test(size_t battles, Eigen::VectorXi soldiers, bool optimistic, bool random, size_t t) {

	size_t l = 2;
	size_t T0 = 100;
	double tol = 0.1;

	Eigen::VectorXd w(battles);

	if (random) {
		//std::srand((unsigned int)time(0));
		w = 0.5 * Eigen::VectorXd::Random(battles).array() + 0.5;
	}
	else {
		w = Eigen::VectorXd::Constant(battles, 1);
	}


	double beta = 0.95;

	auto begin = std::chrono::high_resolution_clock::now();
	auto test = CB(t, l, battles, soldiers, w, beta, T0, tol, optimistic);
	int it = test.run();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin);
	std::filesystem::create_directory("results");

	std::string file_name = "results//k_" + std::to_string(battles) + "_n_" + std::to_string(soldiers[0]) +
		"_opt_" + std::to_string(optimistic) + "_rand_" + std::to_string(random) + ".txt";

	thread_local std::ofstream f;
	f.open(file_name);

	f << "Battles: " << std::to_string(battles) << '\n';

	f << "Soldiers: [" << std::to_string(soldiers(0)) << ", " << std::to_string(soldiers(1)) << ']' << '\n';

	f << "Weights: [";
	for (auto weight : w) {
		f << std::to_string(weight) << ", ";
	}
	f << ']' << '\n';

	if (it + 1 == t) {
		f << "WARNING: MAX ITERATIONS EXECUTED" << '\n';
	}

	f << "Iterations: " << it << '\n';

	double avg_reg = test.get_regrets().back();

	f << "Average regret: " << std::to_string(avg_reg) << '\n';

	int total_seconds = elapsed.count();
	int hours = total_seconds / 3600;
	total_seconds -= hours * 3600;

	int minutes = total_seconds / 60;
	total_seconds -= minutes * 60;

	f << "Runtime: " << std::to_string(hours) << "h "
					 << std::to_string(minutes) << "m " 
					 << std::to_string(total_seconds) << "s " << '\n' << '\n';

	f << "Last 3 rounds: " << '\n';

	for (size_t i = it - 2; i <= it; ++i) {
        f << "t = " << i << ": " << '\n';
		for (size_t j = 0; j < l; ++j) {
			for (size_t h = 0; h < battles; ++h) {
				f << test.get_strategies()(j)(i, h) << " ";
			}
			f << '\n';
		}
		f << '\n' << '\n';
	}
	f.close();
}

int main() {
    
	//MAIN TESTS:
	std::vector<bool> tf = { true, false };
	size_t tmax = 12000;
	int max_soldiers = 5000;
	int max_battles = 200;
	std::vector< std::unique_ptr< std::thread > > threads;

	// max number of soldiers
	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 2; ++j) {
			threads.emplace_back(std::make_unique<std::thread>(run_test, 10, Eigen::Vector2i::Constant(2, max_soldiers), tf[i], tf[j], tmax));
			threads.emplace_back(std::make_unique<std::thread>(run_test, 20, Eigen::Vector2i::Constant(2, max_soldiers), tf[i], tf[j], tmax));
			threads.emplace_back(std::make_unique<std::thread>(run_test, 30, Eigen::Vector2i::Constant(2, max_soldiers), tf[i], tf[j], tmax));
		}	
	}

	// max number of battles 
	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 2; ++j) {
			threads.emplace_back(std::make_unique<std::thread>(run_test, max_battles, Eigen::Vector2i::Constant(2, max_battles), tf[i], tf[j], tmax));
			threads.emplace_back(std::make_unique<std::thread>(run_test, max_battles, Eigen::Vector2i::Constant(2, 2 * max_battles), tf[i], tf[j], tmax));
			threads.emplace_back(std::make_unique<std::thread>(run_test, max_battles, Eigen::Vector2i::Constant(2, 3 * max_battles), tf[i], tf[j], tmax));
		}

	}

	for (auto& t : threads) { // join all the threads
		t->join();
	}

	/*
	//TABLE:
	size_t t = 5000;
	size_t l = 2;
	size_t T0 = 100;
	double tol = 0.1;
	double beta = 0.95;
	bool optimistic = false;



	std::vector<size_t> battles = { 10, 15, 20 };
	Eigen::Vector3<Eigen::VectorXd> weights; 
	weights(0) = Eigen::VectorXd::Constant(10, 1); 
	weights(1) = Eigen::VectorXd::Constant(15, 1);
	weights(2) = Eigen::VectorXd::Constant(20, 1);
	

	Eigen::MatrixXi soldiers { {20, 20}, {20, 25}, {20,30}, {25, 25}, {25, 30}, {30, 30} };

	std::vector<size_t> iterations;
	std::vector<long double> times;

	
	std::ofstream f;
	std::string file_name = "table_not_opt_eig.txt";
	f.open(file_name, 'w');
	f << "Battles" << '\t' << "Soldiers" << '\t' << "Iterations" << '\t' << "Run Time (ms)" << '\n';


	int n = 10;
	//std::cout << soldiers.row(0);

	
	for (size_t i = 0; i < battles.size(); ++i) {
		for (size_t j = 0; j < soldiers.rows(); ++j) {
			long double it_avg = 0;
			long double time_avg = 0;
			for (size_t h = 0; h < n; ++h) {
				auto begin = std::chrono::high_resolution_clock::now();
				CB game = CB(t, 2, battles[i], soldiers.row(j), weights(i), beta, T0, tol, optimistic);
				int it = game.run();
				auto end = std::chrono::high_resolution_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

				it_avg += it;
				time_avg += elapsed.count();
			}
			it_avg /= n;
			time_avg /= n;
			f << std::to_string(battles[i]) << '\t' << '[' << std::to_string(soldiers(j, 0)) << ", " << std::to_string(soldiers(j,1)) << ']' << '\t';
			f << std::to_string(it_avg) << '\t' << std::to_string(time_avg) << '\n';
		}
	}
	
	f.close();
	*/
	

	
	// FOR PLOTTING REGRETS
	/*
	size_t t = 5000;
	size_t l = 2;
	size_t k = 10;
	size_t T0 = 100;
	double tol = 0.1;
	bool optimistic = true;
	auto n = Eigen::VectorXi::Constant(2, 20);
	//auto w = Eigen::VectorXd::Constant(k, 1);
	Eigen::VectorXd w{ {1, 1, 5, 1, 1, 1, 1, 1, 1, 5} };

	double beta = 0.95;

	auto begin = std::chrono::high_resolution_clock::now();
	auto test = CB(t, l, k, n, w, beta, T0, tol, optimistic);
	
	int it = test.run();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin);



	std::ofstream f1;
	std::string file_name_1 = "regrets.txt";
	f1.open(file_name_1, 'w');
	for (auto el : test.get_regrets()) {
		f1 << std::to_string(el) << '\n';
	}
	f1.close();

	std::ofstream f2;
	std::string file_name_2 = "times.txt";
	f2.open(file_name_2, 'w');
	for (size_t i = 0; i <= it; ++i) {
		if ((i) % T0 == 0) {
			f2 << std::to_string(i) << '\n';
		}
	}
	f2.close();

	std::ofstream f3;
	std::string file_name_3 = "stats.txt";
	f3.open(file_name_3, 'w');
	f3 << "Battles: " << std::to_string(k) << '\n';
	f3 << "Soldiers: [" << std::to_string(n[0]) << ", " << std::to_string(n[1]) << ']' << '\n';
	f3 << "Iterations: " << it << '\n';
	f3 << "Runtime: " << std::to_string(elapsed.count()) << '\n' << '\n';
	f3 << "Last 3 rounds: " << '\n';
	for (size_t i = it - 2; i <= it; ++i) {
		for (size_t j = 0; j < l; ++j) {
			for (size_t h = 0; h < k; ++h) {
				f3 << test.get_strategies()(j)(i, h) << " ";
			}
			f3 << '\n';
		}
		f3 << '\n' << '\n';
	}
	f3.close();
	*/

	return 0;
}