#include "CB.h"
#include "Part.h"
#include<iostream>
#include<fstream>
#include<string>
#include<chrono>
#include<random>
#include<cmath>
#include<filesystem>
#include<memory>



int main() {

	// if using "optimistic" update rule
	bool optimistic = false;

	// max number of iterations
	size_t tmax = 100000;

	// number of players
	size_t l = 2;

	// learning rate. may want increase closer to 1 for a large number of soldiers
	double beta = 0.995;

	// calculate regret after this many rounds
	size_t t0 = 100;

	// regret tolerance. stop iterations once this regret is achieved
	double tol = -1;

	// initialization factor
	int init_factor = 0;

	// whether to calculate distance to equilibrium. this is very slow and only implemented
	// for the 0/1 loss.
	bool calc_d2e = false;

	// FIGURE 1, S5 // 
	{
		// number of soldiers each player has
		// make sure this has as many elements as there are soldiers (L)!
		int N[] = { 20, 20 };

		// "fixed" battle values
		double W_fixed[] = { 1, 2, 3, 5, 9 };

		// "random" battle values, these values were generated while ago
		double W_rand[] = { 42, 68, 35, 1, 70 };

		// number of battles
		size_t k = 5;

		Eigen::ArrayXi fixed_strategy_fixed(5);
		fixed_strategy_fixed << 0, 1, 2, 5, 12;

		Eigen::ArrayXi fixed_strategy_rand(5);
		fixed_strategy_rand << 3, 7, 3, 0, 7;
		

		// directory to write log file to 
		std::string prefix = "results//fig_1//"; 
		std::filesystem::create_directories(prefix);
		std::string suffix_1 = "th_fixed";
		auto test_1 = CB(tmax, l, k, N, W_fixed, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, fixed_strategy_fixed);
		test_1.run_test(prefix, suffix_1);

		std::string suffix_2 = "th_rand";
		auto test_2 = CB(tmax, l, k, N, W_rand, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, fixed_strategy_rand);
		test_2.run_test(prefix, suffix_2);
	}
	

	// FIGURE 2, S6 // 
	{
		int N_2008[] = { 114, 98 }; //2008. player 1 is the republican
		int N_2008_flipped[] = { 98, 114 }; //2008 flipped, player 2 is the republican
		double W_2008[] = { 9, 27, 5, 5, 20, 4, 21, 3, 13 }; // CO, FL, NM, NV, OH, NH, PA, MO, VA //2008
		size_t k_2008 = 9;

		Eigen::ArrayXi fixed_strategy_rep(k_2008);
		Eigen::ArrayXi fixed_strategy_dem(k_2008);
		fixed_strategy_rep << 13, 11, 8, 7, 28, 6, 31, 0, 10; //rep2008
		fixed_strategy_dem << 8, 12, 5, 7, 22, 6, 16, 3, 19; //democrat2008

		std::string prefix = "results//fig_2//";
		std::filesystem::create_directories(prefix);
		std::string suffix_1 = "both_mwu";
		auto test_1 = CB(tmax, l, k_2008, N_2008, W_2008, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote);
		test_1.run_test(prefix, suffix_1);

		std::string suffix_2 = "rep_fixed";
		auto test_2 = CB(tmax, l, k_2008, N_2008, W_2008, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, fixed_strategy_rep);
		test_2.run_test(prefix, suffix_2);

		std::string suffix_3 = "dem_fixed";
		auto test_3 = CB(tmax, l, k_2008, N_2008_flipped, W_2008, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, fixed_strategy_dem);
		test_3.run_test(prefix, suffix_3);
	}
	
	// FIGURE 3, S7 // 
	{
		int N_2020[] = { 61, 45 }; //2020. player 1 is the republican
		int N_2020_flipped[] = { 45, 61 }; //2020 flipped, player 2 is the republican
		double W_2020[] = { 20, 16, 15, 29, 16, 18, 11, 10 }; //PA, MI, NC, FL, GA, OH, AZ, WI //2020
		size_t k_2020 = 8;

		Eigen::ArrayXi fixed_strategy_rep(k_2020);
		Eigen::ArrayXi fixed_strategy_dem(k_2020);
		fixed_strategy_rep << 14, 10, 8, 12, 3, 7, 5, 2; //rep2020
		fixed_strategy_dem << 12, 8, 6, 6, 4, 3, 3, 3; //democrat2020

		std::string prefix = "results//fig_3//";
		std::filesystem::create_directories(prefix);
		std::string suffix_1 = "both_mwu";
		auto test_1 = CB(tmax, l, k_2020, N_2020, W_2020, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote);
		test_1.run_test(prefix, suffix_1);

		std::string suffix_2 = "rep_fixed";
		auto test_2 = CB(tmax, l, k_2020, N_2020, W_2020, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, fixed_strategy_rep);
		test_2.run_test(prefix, suffix_2);

		std::string suffix_3 = "dem_fixed";
		auto test_3 = CB(tmax, l, k_2020, N_2020_flipped, W_2020, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, fixed_strategy_dem);
		test_3.run_test(prefix, suffix_3);
	}

	// FIGURE 4, S8 // 
	{
		int N[] = { 30, 30 }; 
		double W[] = { 1, 1, 1, 1, 1, 1 }; 
		size_t k = 6;

		std::vector<double> strong = { 0.8, 0.3, 0, 0, -0.3, -0.8 };
		std::vector<double> weak = { 0.1, 0.05, 0, 0, -0.05, -0.1 };

		std::string prefix = "results//fig_4//";
		std::filesystem::create_directories(prefix);
		std::string suffix_1 = "adv_weak";
		auto test_1 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, Eigen::ArrayXi(), weak);
		test_1.run_test(prefix, suffix_1);

		std::string suffix_2 = "adv_strong";
		auto test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote, calc_d2e, Eigen::ArrayXi(), strong);
		test_2.run_test(prefix, suffix_2);
	}
	
	// FIGURES S1-S4 // 
	{
		// number of soldiers each player has
		// make sure this has as many elements as there are soldiers (L)!
		int N[] = { 20, 20 };

		// "fixed" battle values
		double W_fixed[] = { 1, 2, 3, 5, 9 };

		// "random" battle values, these values were generated while ago
		double W_rand[] = { 42, 68, 35, 1, 70 };

		// number of battles
		size_t k = 5;

		std::vector<CB::init_type> init_types = { CB::init_type::uniform, CB::init_type::proportional, CB::init_type::three_halves };
		std::vector<CB::loss_type> winning_rules = { CB::loss_type::zero_one, CB::loss_type::popular_vote, CB::loss_type::electoral_vote };
		std::vector<int> init_factors = { 0, 500, 1000, 5000 };

		std::string prefix_1 = "results//fig_S1_3//";
		std::filesystem::create_directories(prefix_1);
		std::string suffix_1 = "fixed";
		for (auto initialization_type : init_types) {
			for (auto winning_rule : winning_rules) {
				for (auto initialization_factor : init_factors) {
					auto test_1 = CB(tmax, l, k, N, W_fixed, beta, t0, tol, optimistic, initialization_type, initialization_factor, winning_rule);
					test_1.run_test(prefix_1, suffix_1);
				}
			}
		}

		std::string prefix_2 = "results//fig_S2_4//";
		std::filesystem::create_directories(prefix_2);
		std::string suffix_2 = "rand";
		for (auto initialization_type : init_types) {
			for (auto winning_rule : winning_rules) {
				for (auto initialization_factor : init_factors) {
					auto test_1 = CB(tmax, l, k, N, W_rand, beta, t0, tol, optimistic, initialization_type, initialization_factor, winning_rule);
					test_1.run_test(prefix_2, suffix_2);
				}
			}
		}
	}

	// FIGURE S9 // 
	{
		int N[] = { 20, 20 };
		double W_fixed[] = { 1, 2, 3, 5, 9 };
		size_t k = 5;
		bool calc_d2e_true = true;
		size_t tmax_short = 100000;
		std::string prefix = "results//fig_9//";
		std::filesystem::create_directories(prefix);
		std::string suffix = "d2e";
		auto test_1 = CB(tmax, l, k, N, W_fixed, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::zero_one, calc_d2e_true);
		test_1.run_test(prefix, suffix);
	}



	// TABLE 1 // 
	{
		size_t tmax = 20000;
		size_t l = 2;
		size_t T0 = 100;
		double tol = 0.05;
		double beta = 0.95;
		bool optimistic = true;
		std::vector<size_t> battles = { 10, 15, 20 };
		const int rows = 6;
		const int cols = 2;
		int soldiers[rows][cols] = { {20, 20}, {20, 25}, {20,30}, {25, 25}, {25, 30}, {30, 30} };

		std::vector<size_t> iterations;
		std::vector<long double> times;

		std::ofstream f;
		std::string dir_name = "results//table_1//";
		std::string file_name = dir_name + "timing_table.txt";
		std::filesystem::create_directories(dir_name);

		f.open(file_name, 'w');
		f << "Battles" << " & " << "Soldiers" << " & " << "Time Standard Update (s)" << " & " << "Time Optimistic Update(s)" << '\n';
		int n = 1;

		for (size_t i = 0; i < battles.size(); ++i) {
			for (size_t j = 0; j < rows; ++j) {
				long double it_avg = 0;
				long double time_avg_not = 0;
				long double time_avg_opt = 0;
				for (size_t h = 0; h < n; ++h) {
					if (i == 0) {
						bool opt = 0;
						std::cout << 1 + (rand() % 100) << '\n';
						double weights[] = { 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100),
											1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100) };
						auto begin = std::chrono::high_resolution_clock::now();
						CB game = CB(tmax, 2, battles[i], soldiers[j], weights, beta, T0, tol, opt, CB::init_type::uniform, 0, CB::loss_type::zero_one);
						int it = game.run();
						auto end = std::chrono::high_resolution_clock::now();
						auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
						it_avg += it;
						double time_not = elapsed.count() / 1000.0;
						time_avg_not += std::round(time_not / 0.001) * 0.001;

						opt = 1;
						begin = std::chrono::high_resolution_clock::now();
						CB game_2 = CB(tmax, 2, battles[i], soldiers[j], weights, beta, T0, tol, opt, CB::init_type::uniform, 0, CB::loss_type::zero_one);
						it = game_2.run();
						end = std::chrono::high_resolution_clock::now();
						elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
						double time_opt = elapsed.count() / 1000.0;
						time_avg_opt += std::round(time_opt / 0.001) * 0.001;


					}
					else if (i == 1) {
						bool opt = 0;
						double weights[] = { 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100),
											1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100),
											1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100) };
						auto begin = std::chrono::high_resolution_clock::now();
						CB game = CB(tmax, 2, battles[i], soldiers[j], weights, beta, T0, tol, opt, CB::init_type::uniform, 0, CB::loss_type::zero_one);
						int it = game.run();
						auto end = std::chrono::high_resolution_clock::now();
						auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
						it_avg += it;
						double time_not = elapsed.count() / 1000.0;
						time_avg_not += std::round(time_not / 0.001) * 0.001;

						opt = 1;
						begin = std::chrono::high_resolution_clock::now();
						CB game_2 = CB(tmax, 2, battles[i], soldiers[j], weights, beta, T0, tol, opt, CB::init_type::uniform, 0, CB::loss_type::zero_one);
						it = game_2.run();
						end = std::chrono::high_resolution_clock::now();
						elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
						double time_opt = elapsed.count() / 1000.0;
						time_avg_opt += std::round(time_opt / 0.001) * 0.001;
					}
					else {
						bool opt = 0;
						double weights[] = { 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100),
											1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100),
											1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100),
											1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100) };
						auto begin = std::chrono::high_resolution_clock::now();
						CB game = CB(tmax, 2, battles[i], soldiers[j], weights, beta, T0, tol, opt, CB::init_type::uniform, 0, CB::loss_type::zero_one);
						int it = game.run();
						auto end = std::chrono::high_resolution_clock::now();
						auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
						it_avg += it;
						double time_not = elapsed.count() / 1000.0;
						time_avg_not += std::round(time_not / 0.001) * 0.001;

						opt = 1;
						begin = std::chrono::high_resolution_clock::now();
						CB game_2 = CB(tmax, 2, battles[i], soldiers[j], weights, beta, T0, tol, opt, CB::init_type::uniform, 0, CB::loss_type::zero_one);
						it = game_2.run();
						end = std::chrono::high_resolution_clock::now();
						elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
						double time_opt = elapsed.count() / 1000.0;
						time_avg_opt += std::round(time_opt / 0.001) * 0.001;
					}

				}
				it_avg /= n;
				time_avg_not /= n;
				time_avg_opt /= n;
				f << std::to_string(battles[i]) << " " << '&' << "{[}"
					<< std::to_string(soldiers[j][0]) << ", " << std::to_string(soldiers[j][1]) << "{]}" << " & ";

				f << std::to_string(time_avg_not) << " & " << std::to_string(time_avg_opt) << '\n';
			}
		}
		f.close();
	}
	
	
	// FIGURE S10 //
	{
		int N[] = { 20, 20 };
		double beta = 0.995;
		int t_max = 100000;
		double W_10[10];
		size_t k_10 = 10;
		size_t init_factor = 0;

		std::random_device rd;  // Obtain a random number from hardware
		std::mt19937 gen(rd()); // Seed the generator
		std::uniform_int_distribution<> distrib(1, 100); // Define the range [1, 100]

		for (double& num : W_10) {
			num = distrib(gen);
		}

		bool optimistic_options[] = { false, true };
		std::vector<CB::loss_type> winning_rules = { CB::loss_type::zero_one, CB::loss_type::popular_vote, CB::loss_type::electoral_vote };
		for (bool optimistic_option : optimistic_options) {
			for (auto winning_rule : winning_rules) {
				std::string prefix = "results//fig_S10//";
				std::filesystem::create_directories(prefix);
				std::string suffix = "opt_tests";
				auto test_1 = CB(tmax, l, k_10, N, W_10, beta, t0, tol, optimistic_option, CB::init_type::uniform, init_factor, winning_rule);
				test_1.run_test(prefix, suffix);
			}
		}

	}
	
	return 0;
}
