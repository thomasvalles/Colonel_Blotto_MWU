#pragma once
#include<vector>
#include<algorithm>
#include<cmath>
#include<numeric>
#include<random>
#include<Eigen>
#include<time.h>
#include<chrono>
#include<filesystem>
#include<fstream>


typedef Eigen::Vector< long double, Eigen::Dynamic                > Vecld;
typedef Eigen::Array< long double, Eigen::Dynamic, 0               > Arld;
typedef Eigen::Array< long double, Eigen::Dynamic, Eigen::Dynamic > Arrld;

/**
@class CB represents a Colonel Blotto game
*/
class CB {
public:
    
    enum init_type { uniform = 0, proportional = 1, three_halves = 2 };
    enum loss_type { zero_one = 0, popular_vote = 1, electoral_vote = 2, ev_adj = 3 };
    /**
    Game constructor, initializes strategies
    @param T_: Number of rounds to run the game
    @param L_: Number of players
    @param N_: Vector containing number of soldiers for each player
    @param W_: Vector containing weights for each battle
    @param beta_: Learning rate in (0, 1)
    @param T0_: Calculate the regret every T0 rounds
    @param optimistic: True if running optimistic rwm
    */
    CB(size_t _T, size_t _L, size_t _k, int _N[], double _W[], 
        double _beta, size_t _T0, double _tol, bool _optimistic, 
        init_type _init, size_t _init_factor, loss_type _lt);

    /**
    Simulates the game using the specified parameters
    @return The number of rounds played
    */
    int run();

    /**
    Public strategies getter
    */

    auto get_strategies() {
        return strategies;
    }

    /*
    std::vector<std::vector<std::vector<int>>> get_strategies() {
        std::vector<std::vector<std::vector<int>>> strats(TMAX, std::vector<std::vector<int>>(L, std::vector<int>(k)));
        for (size_t i = 0; i < TMAX; ++i) {
            for (size_t j = 0; j < L; ++j) {
                for (size_t x = 0; x < k; ++x) {
                    strats[i][j][x] = strategies(j)(i, x);
                }
            }
        }
        return strats;
    }
    */

    /**
    Public regrets getter
    */
    auto get_regrets() {
        return regrets;
    }

    Arrld get_weights() {
        return dist;
    }

    Eigen::ArrayXXd get_avg_al() {
        return avg_al;
    }


    void run_test(); 


private:

    size_t TMAX; size_t L; size_t k; size_t T0; double tol; bool optimistic; double beta; size_t init_factor;
    Eigen::VectorXi N; Eigen::VectorXd W; Eigen::VectorXi s0; init_type init; loss_type lt; Arrld dist;
    size_t numerical_correction; int sum_of_values; std::vector<double> learner_cum_loss = {0, 0};
    std::vector<double> reward_of_avg = { 0, 0 };
    
    int get_sum() {
        return sum_of_values;
    }

   

    // 3-d vector containing the strategy at each [time][player][battlefield]
    //std::vector<std::vector<std::vector<int>>> strategies;
    Eigen::Vector<Eigen::ArrayXXi, Eigen::Dynamic> strategies;

    // 3-d vector containing the historical loss for each [player][battle][strategy]
    //std::vector<std::vector<std::vector<long double>>> hist_loss;
    Eigen::Vector<Arrld, Eigen::Dynamic> hist_loss;
    Eigen::Vector<Arrld, Eigen::Dynamic> hist_reward;
    Eigen::Vector<Arrld, Eigen::Dynamic> actual_loss;
    Eigen::Vector<Arrld, Eigen::Dynamic> initialized_loss;
    Eigen::ArrayXXd avg_al;


    //  sum_i (regret of player i) / t, calculated ever T0 rounds
    std::vector<std::vector<long double>> regrets;
    std::vector<std::vector<long double>> eq_dis;


    /**
    Initialize historical loss matrix based on init_type parameter
    */
    Eigen::Vector<Arrld, Eigen::Dynamic> initialize_loss();


    /**
    assigns strategy to player j in round i
    */
    void assign_strat(int i, int j);

    /**
    Updates the historical loss vector
    @param time: The round to add to the historical loss
    */
    void update_hist_loss(size_t time);

    /**
    Executes the rwm algorithm
    @param t: Current round
    @param l: Player
    @return: Strategy vector- number of soldiers to allocate to each battlefield
    */
    Eigen::VectorXi rwm(size_t t, size_t l);

    /**
    Computes the partition function f
    @param t: Current round
    @param l: Player
    @return Partition function evaluated at [battlefield][strategy]
    */
    Arrld get_partition(size_t t, size_t l);

    /**
    Gets the loss of a single strategy in a battlefield
    @param t: Round
    @param l: Player
    @param h: Battle
    @param x: Strategy (number of soldiers allotted)
    @return: The loss
    */
    Arrld get_loss(size_t t, size_t l);


    double get_reward_vec(Eigen::ArrayXi x, Eigen::ArrayXi y);

    double get_reward_num(int x, int y, size_t h);

    Arrld get_reward(size_t t, size_t l);

    double calculate_regret(size_t time);

    double calculate_distance_to_eq(size_t time);

    double distance_helper(size_t player, size_t time);

    /**
    Gets the best historical loss according to the historical loss matrix
    @param l: Player
    @return: The best historical loss
    */
    long double get_best_hist_loss(size_t time, size_t l, Arrld mat);


    // Some math functions

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

    static double incbeta(double a, double b, double x) {
        const long double STOP = 1.0e-8;
        const long double TINY = 1.0e-30;
        if (x < 0.0 || x > 1.0)  return std::numeric_limits<double>::quiet_NaN();

        /*The continued fraction converges nicely for x < (a+1)/(a+b+2)*/
        if (x > (a + 1.0) / (a + b + 2.0)) {
            return (1.0 - incbeta(b, a, 1.0 - x)); /*Use the fact that beta is symmetrical.*/
        }

        /*Find the first part before the continued fraction.*/
        const double lbeta_ab = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
        const double front = std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta_ab) / a;

        /*Use Lentz's algorithm to evaluate the continued fraction.*/
        double f = 1.0, c = 1.0, d = 0.0;

        int i, m;
        for (i = 0; i <= 200; ++i) {
            m = i / 2;

            double numerator;
            if (i == 0) {
                numerator = 1.0; /*First numerator is 1.0.*/
            }
            else if (i % 2 == 0) {
                numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)); /*Even term.*/
            }
            else {
                numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1)); /*Odd term.*/
            }

            /*Do an iteration of Lentz's algorithm.*/
            d = 1.0 + numerator * d;
            if (std::fabs(d) < TINY) d = TINY;
            d = 1.0 / d;

            c = 1.0 + numerator / c;
            if (std::fabs(c) < TINY) c = TINY;

            const double cd = c * d;
            f *= cd;

            /*Check for stop.*/
            if (std::fabs(1.0 - cd) < STOP) {
                return front * (f - 1.0);
            }
        }

        return  std::numeric_limits<double>::quiet_NaN(); /*Needed more loops, did not converge.*/
    }



};