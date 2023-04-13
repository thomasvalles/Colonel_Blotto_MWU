#pragma once
#include<vector>
#include<algorithm>
#include<cmath>
#include<numeric>
#include<random>
#include<Eigen>
#include<time.h>

typedef Eigen::Vector< long double, Eigen::Dynamic                > Vecld;
typedef Eigen::Array< long double, Eigen::Dynamic, Eigen::Dynamic > Arrld;

/**
@class CB represents a Colonel Blotto game
*/
class CB {
public:

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
    CB(size_t _T, size_t _L, size_t _k, Eigen::VectorXi _N, Eigen::VectorXd _W, double _beta, size_t _T0, double _tol, bool _optimistic);

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

    /**
    Public regrets getter
    */
    auto get_regrets() {
        return regrets;
    }


private:

    size_t TMAX;
    size_t L;
    size_t k;
    size_t T0;
    double tol;
    bool optimistic;
    Eigen::VectorXi N;
    Eigen::VectorXd W;
    double beta;


    // 3-d vector containing the strategy at each [time][player][battlefield]
    //std::vector<std::vector<std::vector<int>>> strategies;
    Eigen::Vector<Eigen::ArrayXXi, Eigen::Dynamic> strategies;

    // 3-d vector containing the historical loss for each [player][battle][strategy]
    //std::vector<std::vector<std::vector<long double>>> hist_loss;
    Eigen::Vector<Arrld, Eigen::Dynamic> hist_loss;


    //  sum_i (regret of player i) / t, calculated ever T0 rounds
    std::vector<double> regrets;

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

    /**
    Gets the best historical loss according to the historical loss matrix
    @param l: Player
    @return: The best historical loss
    */
    long double get_best_hist_loss(size_t time, size_t l);


};