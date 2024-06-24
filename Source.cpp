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

	// number of battles
	const int array_size = 5;
	size_t k = array_size;

	// calculate regret after this many rounds
	size_t t0 = 100;

	// regret tolerance. stop iterations once this regret is achieved
	double tol = 0.05;

	// learning rate. may want increase closer to 1 for a large number of soldiers
	double beta = 0.95;

	// number of soldiers each player has
	// make sure this has as many elements as there are soldiers (L)!
	int N[] = { 20, 20 };

	// battlefield weights, for now, just set to random integers
	// make sure this has as many elements as battlefields (k)!
	double W[array_size];

	// Seed the random number generator
	//srand(static_cast<unsigned>(time(0)));

	// Generating random integers from 1 to 100 and storing them in the array
	for (int i = 0; i < array_size; ++i) {
		W[i] = rand() % 100 + 1; // Random integer between 1 and 100
	}


	// type of initialization, one of: uniform, proportional, three_halves,
	CB::init_type initialization_type = CB::init_type::uniform;

	// doing no initialization
	size_t initialization_factor = 0;

	// loss type, one of: zero_one, popular_vote, electoral_vote, or ev_adj
	CB::loss_type winning_rule = CB::loss_type::zero_one;

	// can use a fixed strategy, if you don't want player a to use a fixed strategy,
	// you can either leave this as an empty array or just not provide the argument in the constructor.
	Eigen::ArrayXi fixed_strategy(0);
	//fixed_strategy << 4, 4, 4, 4, 4;

	//srand(time(NULL));

	// directory to write log file to 
	std::string prefix = "C://Users//tom13//Desktop//FA21//cb//results//";

	// test postfix
	std::string suffix = "0623_testb";
	auto test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, initialization_type, initialization_factor, winning_rule, fixed_strategy);
	test_2.run_test(prefix, suffix);








	/*CODE FOR RUNNING SOME OF THE EXPERIMENTS USED IN THE PAPER*/



/**
Runs a specified test. Outputs test results to file named by parameters
@param battles: the number of battles to run
@param soldiers: vector of soldiers for the two players
@param optimistic: true if optimistic hedge
@param random: true of weights are to be randomly selected
*/
/*
void run_test(size_t battles, Eigen::VectorXi soldiers, bool optimistic, bool random, size_t t) {


	Eigen::VectorXd electoral_weights(51);
	electoral_weights << 45, 41, 27, 26, 26, 25, 21, 17, 17, 14,
		13, 13, 12, 12, 12, 11, 10, 10, 10, 10,
		9, 9, 9, 8, 8, 8, 8, 7, 7, 7,
		6, 6, 6, 6, 5, 4, 4, 4, 4, 4,
		4, 4, 4, 4, 3, 3, 3, 3, 3, 3,
		3;

	size_t l = 2;
	size_t T0 = 100;
	double tol = 0.01;

	Eigen::VectorXd w(battles);

	if (random) {
		//std::srand((unsigned int)time(0));
		w = 0.5 * Eigen::VectorXd::Random(battles).array() + 0.5;
	}
	else {
		w = electoral_weights;
		//w = Eigen::VectorXd::Constant(battles, 1);
	}


	double beta = 0.95;

	auto begin = std::chrono::high_resolution_clock::now();
	auto test = CB(t, l, battles, soldiers, w, beta, T0, tol, optimistic);
	int it = test.run();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin);
	std::filesystem::create_directory("results");

	//std::string file_name = "results//k_" + std::to_string(battles) + "_n_" + std::to_string(soldiers[0]) +
	//	"_opt_" + std::to_string(optimistic) + "_rand_" + std::to_string(random) + ".txt";
	std::string file_name = "results//k_" + std::to_string(battles) + "_n_" + std::to_string(soldiers[0]) +
		"_opt_" + std::to_string(optimistic) + "_electoral_" + std::to_string(random) + ".txt";

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
*/

//double W[] = { 1, 5};
//double W[] = { 1, 2, 3, 5, 9 };
//double W[] = { 42, 68, 35, 1, 70 };
//double W[] = { 1, 9};
//double W[] = { 10, 10, 10, 10, 10, 10 };
//double W[] = { 16, 16, 11, 15, 10, 19 }; //NC, GA, AZ, MI, WI, PA
//auto w = 0.5 * Eigen::VectorXd::Random(k).array() + 0.5;

//int N[] = { 45, 61 }; //2020
//int N[] = { 114, 98 }; //2008

//double W[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

//double W[] = { 1, 2, 3, 5, 9 };
//double W[] = {42, 68, 35, 1, 70};
//double W[] = { 20, 16, 15, 29, 16, 18, 11, 10 }; //PA, MI, NC, FL, GA, OH, AZ, WI //2020
//double W[] = { 9, 27, 5, 5, 20, 4, 21, 3, 13 }; // CO, FL, NM, NV, OH, NH, PA, MO, VA //2008

//double W[] = { 1, 1, 1, 1, 1 };

//double W[] = { 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100),  1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100), 1 + (rand() % 100) };

//auto test_1 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, 0, CB::loss_type::ev_adj);
//test_1.run_test();


//auto test_3 = CB(tmax, l, k, N, W, beta, t0, tol, !optimistic, CB::init_type::uniform, 0, CB::loss_type::popular_vote);
//test_3.run_test();
//auto test_4 = CB(tmax, l, k, N, W, beta, t0, tol, !optimistic, CB::init_type::uniform, 0, CB::loss_type::electoral_vote);
//test_4.run_test();

/*

test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, 0, CB::loss_type::zero_one);
test_2.run_test();

test_3 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, 0, CB::loss_type::popular_vote);
test_3.run_test();

test_4 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, 0, CB::loss_type::electoral_vote);
test_4.run_test();

*/

/*{45, 41, 27, 26, 26, 25, 21, 17, 17, 14,
	13, 13, 12, 12, 12, 11, 10, 10, 10, 10,
	9, 9, 9, 8, 8, 8, 8, 7, 7, 7,
	6, 6, 6, 6, 5, 4, 4, 4, 4, 4,
	4, 4, 4, 4, 3, 3, 3, 3, 3, 3,
	3};
	*/

	int s0[] = { 45, 41, 27, 26, 26, 25, 21, 17, 17, 14,
		13, 13, 12, 12, 12, 11, 10, 10, 10, 10,
		9, 9, 9, 8, 8, 8, 8, 7, 7, 7,
		6, 6, 6, 6, 5, 4, 4, 4, 4, 4,
		4, 4, 4, 4, 3, 3, 3, 3, 3, 3,
		3 };
	int s_three_halves[] = { 74, 64, 34, 32, 33, 31, 24, 17, 17, 13,
		12, 12, 10, 10, 10, 9, 8, 8, 8, 8,
		7, 7, 7, 6, 6, 6, 6, 5, 5, 5,
		4, 4, 4, 4, 3, 2, 2, 2, 2, 2,
		2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
		1 };

	/*
	int N1[] = { 30, 30 };
	double W1[] = { 1, 2, 2, 4, 1,
					1, 2, 2, 4, 1,
					1, 2, 2, 4, 1,
					1, 2, 2, 4, 1};
	CB test_1 = CB(tmax, l, 20, N1, W1, beta, t0, 0.01, optimistic, CB::init_type::uniform, 0, CB::loss_type::zero_one);
	test_1.run_test();
	*/



	for (size_t i = 0; i < 1; ++i) {
		//optimistic = (i == 0) ? true : false;

		size_t init_factor = 500;

		/*
		CB test_1 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::zero_one);
		test_1.run_test();

		CB test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::zero_one);
		test_2.run_test();

		CB test_3 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::zero_one);
		test_3.run_test();


		CB test_4 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::popular_vote);
		test_4.run_test();

		CB test_5 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::popular_vote);
		test_5.run_test();

		CB test_6 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::popular_vote);
		test_6.run_test();


		CB test_7 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote);
		test_7.run_test();

		CB test_8 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::electoral_vote);
		test_8.run_test();

		CB test_9 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::electoral_vote);
		test_9.run_test();
		*/

		//******************************************************************************

		init_factor = 1000;

		/*

		test_1 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::zero_one);
		test_1.run_test();

		test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::zero_one);
		test_2.run_test();

		test_3 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::zero_one);
		test_3.run_test();

		test_4 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::popular_vote);
		test_4.run_test();

		test_5 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::popular_vote);
		test_5.run_test();

		test_6 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::popular_vote);
		test_6.run_test();


		test_7 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote);
		test_7.run_test();

		test_8 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::electoral_vote);
		test_8.run_test();

		test_9 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::electoral_vote);
		test_9.run_test();
		*/

		//******************************************************************************

		init_factor = 2000;
		/*
		test_1 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::zero_one);
		test_1.run_test();

		test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::zero_one);
		test_2.run_test();

		test_3 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::zero_one);
		test_3.run_test();

		test_4 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::popular_vote);
		test_4.run_test();

		test_5 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::popular_vote);
		test_5.run_test();

		test_6 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::popular_vote);
		test_6.run_test();

		test_7 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote);
		test_7.run_test();

		test_8 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::electoral_vote);
		test_8.run_test();

		test_9 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::electoral_vote);
		test_9.run_test();

		*/
		//******************************************************************************
		init_factor = 5000;

		/*

		test_1 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::zero_one);
		test_1.run_test();

		test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::zero_one);
		test_2.run_test();

		test_3 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::zero_one);
		test_3.run_test();

		test_4 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::popular_vote);
		test_4.run_test();

		test_5 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::popular_vote);
		test_5.run_test();

		test_6 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::popular_vote);
		test_6.run_test();



		test_7 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote);
		test_7.run_test();

		test_8 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::proportional, init_factor, CB::loss_type::electoral_vote);
		test_8.run_test();

		test_9 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::three_halves, init_factor, CB::loss_type::electoral_vote);
		test_9.run_test();
		*/

		/***************************************************************************/
		init_factor = 0;

		/*
		test_1 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::zero_one);
		test_1.run_test();

		test_2 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::popular_vote);
		test_2.run_test();

		CB test_3 = CB(tmax, l, k, N, W, beta, t0, tol, optimistic, CB::init_type::uniform, init_factor, CB::loss_type::electoral_vote);
		test_3.run_test();

		*/
	}


	//run_test(battles, Eigen::Vector2i::Constant(2, soldiers), optimistic, random, tmax);


	/*
	//MAIN TESTS:
	std::vector<bool> tf = { true, false };
	size_t tmax = 12000;
	int max_soldiers = 50;
	int max_battles = 10;
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
*/

//TABLE:
/*
size_t tmax = 20000;
size_t l = 2;
size_t T0 = 100;
double tol = 0.05;
double beta = 0.95;
bool optimistic = true;



std::vector<size_t> battles = { 10, 15, 20 };
//Eigen::Vector3<double[]> weights = { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } ,
//{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
//{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };


	const int rows = 6;
	const int cols = 2;
	int soldiers[rows][cols] = { {20, 20}, {20, 25}, {20,30}, {25, 25}, {25, 30}, {30, 30} };

	std::vector<size_t> iterations;
	std::vector<long double> times;


	std::ofstream f;
	std::string file_name = "C://Users//tom13//Desktop//FA21//cb//table_new.txt";
	f.open(file_name, 'w');
	f << "Battles" << " & " << "Soldiers" << " & " << "Time Standard Update (s)" << " & "  << "Time Optimistic Update(s)" << '\n';
	//double asdf[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	//CB game = CB(tmax, 2, battles[0], soldiers[0], asdf, beta, T0, tol, optimistic, CB::init_type::uniform, 0, CB::loss_type::zero_one);
	//game.run_test();
	int n = 1;
	//std::cout << soldiers.row(0);


	for (size_t i = 0; i < battles.size(); ++i) {
		for (size_t j = 0; j < rows; ++j) {
			long double it_avg = 0;
			long double time_avg_not = 0;
			long double time_avg_opt = 0;
			//double weights[10];
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
					time_avg_not += std::round(time_not / 0.001) * 0.001 ;

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
					double time_not = elapsed.count() / 1000.0 ;
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
				<< std::to_string(soldiers[j][0]) << ", " << std::to_string(soldiers[j][1]) << "{]}" << " & " ;

			f << std::to_string(time_avg_not) << " & " << std::to_string(time_avg_opt) << '\n';
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
