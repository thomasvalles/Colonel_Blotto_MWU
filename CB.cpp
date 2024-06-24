#include "CB.h"
#include "Part.h"

CB::CB(size_t _T, size_t _L, size_t _k, int _N[], double _W[], 
	double _beta, size_t _T0, double _tol, bool _optimistic,
	init_type _init, size_t _init_factor, loss_type _lt, Eigen::ArrayXi _fixed_strategy) {

	//if you want no numerical correction, set numerical_correction = TMAX;
	TMAX = _T; L = _L; k = _k; beta = _beta; T0 = _T0; tol = _tol; optimistic = _optimistic;
	init = _init; lt = _lt; init_factor = _init_factor; numerical_correction = TMAX + 2; 
	regrets  = std::vector<std::vector<long double>>(  (int) (TMAX / T0) + 1, std::vector<long double>(4));
	//eq_dis = std::vector<std::vector<long double>>((int)(TMAX / T0) + 1, std::vector<long double>(4));

	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 generator(1234);
	gen = generator;// ^ (static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()))); // Standard mersenne_twister_engine seeded with rd()


	fixed_strategy = _fixed_strategy;
	if (fixed_strategy.size() > 0){
		if (fixed_strategy.size() != k) {
			std::cout << "Fixed strategy size: " << fixed_strategy.size() << " . Expected number of battlefields: " << k << '\n';
		}
		if (fixed_strategy.sum() != _N[0]) {
			std::cout << "Fixed strategy sum: " << fixed_strategy.sum() << " . Expected number of soldiers: " << _N[0] <<'\n';
		}
	}


	N = Eigen::Map<Eigen::VectorXi>(_N, L);
	W = Eigen::Map<Eigen::VectorXd>(_W, k);
	//std::srand((unsigned int)time(0));
	//std::srand(1);

	//(player)(time, battlefield)
	strategies = Eigen::Vector<Eigen::ArrayXXi, Eigen::Dynamic>(L);

	//(player)(battlefield, amount)
	hist_loss = Eigen::Vector<Arrld, Eigen::Dynamic>(L);
	//hist_reward = Eigen::Vector<Arrld, Eigen::Dynamic>(L);

	dist = Arrld::Zero(k, N(0) + 1);
	avg_al = Eigen::ArrayXXd::Zero(2, k);
	for (size_t i = 0; i < L; ++i) {
		strategies(i) = Eigen::ArrayXXi::Zero(TMAX + 1, k);
		hist_loss(i) = Arrld::Zero(k, N(i) + 1);
		//hist_reward(i) = Arrld::Zero(k, N(i) + 1);
	}
	actual_loss = hist_loss;
	sum_of_values = W.sum();
	W = W / sum_of_values;
	initialized_loss = initialize_loss();
	//W.normalize();
}

Eigen::Vector<Arrld, Eigen::Dynamic> CB::initialize_loss() {
	auto initialized_loss = Eigen::Vector<Arrld, Eigen::Dynamic>(L);
	for (size_t i = 0; i < L; ++i) {
		initialized_loss(i) = Arrld::Zero(k, N(i) + 1);
	}
	if (init_factor != 0) {
		Eigen::ArrayXd comparison_vec;
		for (size_t l = 0; l < L; ++l) {
			if (init == init_type::uniform) {
				comparison_vec = N(1 - l) * Eigen::ArrayXd::Constant(k, 1 / (double)k);
				//std::cout << comparison_vec << '\n';
			}
			else if (init == init_type::proportional) {
				comparison_vec = (W * N(1 - l)).array();
			}
			else if (init == init_type::three_halves) {
				Eigen::ArrayXd values_arr = W.array();
				auto th = Eigen::pow(values_arr, 1.5);
				comparison_vec = th / th.sum();
				//std::cout << comparison_vec << '\n' << '\n';
			}

			Arrld result = Arrld::Zero(k, N(l) + 1);;

			auto values = W.replicate(1, N(l) + 1).array();
			if (lt == loss_type::zero_one) {
				auto m_greater = comparison_vec.replicate(N(l) + 1, 1).transpose()
		> Eigen::ArrayXd::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose();
				auto casted_m_greater = m_greater.cast<double>();
				auto m_equal = comparison_vec.replicate(N(l) + 1, 1).transpose()
					== Eigen::ArrayXd::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose();
				auto casted_m_equal = m_equal.cast<double>();

				result = ((double)init_factor * values * casted_m_greater
					+ ((double)1 / L) * (double)init_factor * values * casted_m_equal).cast<long double>();
			}

			else if (lt == loss_type::popular_vote) {
				Eigen::ArrayXXd m1 = comparison_vec.replicate(N(l) + 1, 1).transpose().cast<double>();
				Eigen::ArrayXXd m2 = Eigen::ArrayXi::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose().cast<double>();
				Eigen::ArrayXXd m3 = m1 + m2;
				//Eigen::ArrayXXd m3 = m1 + m2 + Eigen::ArrayXXd::Constant(k, N(l) + 1, 0.0001);
				result = ((double)init_factor * values * m1.binaryExpr(m3, [](auto x, auto y) { return y == 0 ? 0.5 : x / y; })).cast<long double>();
				//loss = (values * (m1 / m3)).cast<long double>();
			}

			else if (lt == loss_type::electoral_vote) {
				for (size_t h = 0; h < k; ++h) {
					for (size_t x = 0; x <= N(l); ++x) {
						int proportionality = 10; //must be even
						int nh = W(h) * sum_of_values * proportionality; // proportional to the value of each battle 

						//P(X <= k) = 1 - incbeta(k+1, n-k), we want P(X >= k) = 1 - bincdf(k-1, n, p) = incbeta(k, n-(k-1)) 
						//k = (n / 2) + 1 for player 0 (player 1 must win majority) k = (n / 2) for player 1 (player 0 wins in tie)
						//p = opponent allocation / (opponent allocation + your proposed allocation)
						long double p;
						if (comparison_vec(h) == 0 && x == 0) {
							result(h, x) = W(h) * 0.5;
						}
						else {
							p = 1. * x / (comparison_vec(h) + x);
							int kappa = (nh / 2) - 1;
							double cum_pr_half_minus_one = 1 - incbeta(kappa + 1, nh - kappa, p);
							double pr_half = (1 - incbeta(kappa + 2, nh - (kappa + 1), p)) - cum_pr_half_minus_one;
			
							result(h, x) = (double)init_factor * W(h) * (cum_pr_half_minus_one + 0.5 * pr_half);
						}
					}
				}
			}

			hist_loss(l) = result;
			initialized_loss(l) = result;
		}


	}
	return initialized_loss;
}

void CB::assign_strat(int i, int j) {
	strategies(j).row(i) = rwm(i, j);
}

int CB::run() {
	int i = 0;
	
	do {
		//for each player
		
		for (int j = 0; j < L; ++j) {
			if (fixed_strategy.size() > 0) {
				if (j == 0){
					strategies(j).row(i) = fixed_strategy;
				}
				else {
					strategies(j).row(i) = this->rwm(i, j);
				}
			}
			else {
				strategies(j).row(i) = this->rwm(i, j);
			}
			//auto assign_strat = [this](int ind1, int ind2) {
			//	strategies(ind2).row(ind1) = this->rwm(ind1, ind2);
			//};
			/*
			if (j == 0) {//player 0 plays 3/2's strategy deterministically
				//strategies(j).row(i) = rwm(i, j);
				//Eigen::Array<int, 8, 1> strat{ 12, 8, 6, 6, 4, 3, 3, 3 }; //democrat2020
				//Eigen::Array<int, 8, 1> strat{ 14, 10, 8, 12, 3, 7, 5, 2 }; //rep2020
				//Eigen::Array<int, 9, 1> strat{ 8, 12, 5, 7, 22, 6, 16, 3, 19 }; //democrat2008
				Eigen::Array<int, 9, 1> strat{ 13, 11, 8, 7, 28, 6, 31, 0, 10 }; //rep2008

				strategies(j).row(i) = strat;
				//strategies(j).row(i) = rwm(i,j);
				/*
				double s_th = 0;
				int used = 0;
				for (size_t h = 0; h < k; ++h) {
					s_th += std::pow(W(h), 1.5);
				}
				for (size_t h = 0; h < k - 1; ++h) {
					strategies(j)(i, h) = (int)(N(j) * std::pow(W(h), 1.5) / s_th + 0.5);
					used += strategies(j)(i, h);
				}

				strategies(j)(i, k - 1) = N(j) - used;
				
			}
			else {//otherwise do rwm
				strategies(j).row(i) = rwm(i, j);
			}
			
			*/
			
			//auto threadFunction = std::make_unique<std::function<void()>>([CB::assign_strat](i, j) {
            //assign_strat(i, j);
			// });
			//assign_strat(i, j);
			//threads.emplace_back(assign_strat, i, j);
			
		}
		
		avg_al.row(0) += strategies(0).row(i).cast<double>();
		avg_al.row(1) += strategies(1).row(i).cast<double>();
		for (size_t h = 0; h < k; ++h) { //for each battle
			dist(h, strategies(0)(i, h)) += 1; //+1 if allocated that many soldiers
		}

		update_hist_loss(i);
		
		// every T0 rounds, calculate the regret 
		if ((i) % T0 == 0) {
			calculate_regret(i);
			//calculate_distance_to_eq(i);
		}
		++i;

		
	} while ((i <= TMAX) && (regrets[(i -1) / T0 ].back() > tol));
	
	dist = dist / (i - 1);
	avg_al = avg_al / (i - 1);
	
	return (i - 1);
}

double CB::calculate_distance_to_eq(size_t time) {
	
	double p0_dist = distance_helper(0, time);
	double p1_dist = distance_helper(1, time);
	double d_eq = std::max(p0_dist, p1_dist);
	eq_dis[int(time / T0)] = std::vector<long double>{ (double)time, p0_dist, p1_dist, d_eq };

	return d_eq;
	
}

double CB::distance_helper(size_t player, size_t time) {

	long double best_resp_to_avg = -get_best_hist_loss(time, player, -hist_reward(player).transpose()) / ((double)time + 1);
	return best_resp_to_avg - reward_of_avg[player] / (((double)time + 1) * ((double)time + 1)) ;
}

//represents x^T C y
double CB::get_reward_vec(Eigen::ArrayXi x, Eigen::ArrayXi y) {
	double loss = 0;
	Eigen::ArrayXd values = W.array().cast<double>();

	if (lt == loss_type::zero_one) {
		Eigen::ArrayXd greater = (x.array() > y.array()).cast<double>();
		loss += (greater * values).sum();

		Eigen::ArrayXd eq = (x.array() == y.array()).cast<double>();
		loss += (eq * values / 2).sum();
	}
	return loss;
}

double CB::get_reward_num(int x, int y, size_t h) {
	long double loss = 0;
	if (lt == loss_type::zero_one) {
		loss += (long double)(x > y) * W(h) + (long double)(x == y) * W(h) / 2.0;
	}
	return loss;
}


void CB::calculate_regret(size_t time) {
	double window_length;
	if (time < numerical_correction) {
		window_length = (int)time;
	}
	else {
		window_length = (int)numerical_correction;
	}
	long double regret = 0;
	long double r1 = 0;
	long double r2 = 0;
	for (size_t l = 0; l < L; ++l) {
		Arrld loss_mat;
		if (optimistic) {
			loss_mat = (hist_loss(l) - get_loss(time, l) - initialized_loss(l)).transpose();
		}
		else {
			loss_mat = (hist_loss(l) - initialized_loss(l)).transpose();
		}
		long double best_hist_loss = get_best_hist_loss(time, l, loss_mat);

		if (l == 0) {
			r1 = ((learner_cum_loss[l] / (window_length + 1)) - (best_hist_loss / (window_length + 1)));
		}
		else {
			r2 = ((learner_cum_loss[l] / (window_length + 1)) - (best_hist_loss / (window_length + 1)));
		}
	}

	regrets[int (time / T0)] = std::vector<long double>{ (double)time, r1, r2, r1+r2 };
}

void CB::update_hist_loss(size_t time) {
	
	// for each player
	for (size_t l = 0; l < L; ++l) {
		auto loss = get_loss(time, l);

		// for each battle
		for (size_t h = 0; h < k; ++h) {
			learner_cum_loss[l] += loss(h, strategies(l)(time, h));
		}
	
		//hist_reward(l) += get_reward(time - 1, l);

		if (optimistic) {
			if (time > 0) {
				actual_loss(l) += 2 * loss - get_loss(time - 1, l);
				if (time < numerical_correction) {
					hist_loss(l) += 2 * loss - get_loss(time - 1, l);
				}
				else {
					hist_loss(l) += 2 * loss - get_loss(time - 1, l) - get_loss(time - numerical_correction, l);
				}
			}
			else {
				actual_loss(l) += 2 * loss;
				hist_loss(l) += 2 * loss;
			}
		}
		else {
			actual_loss(l) += loss;
			if (time < numerical_correction) {
				hist_loss(l) += loss;
			}
			else {
				hist_loss(l) += loss - get_loss(time - numerical_correction, l);
			}

		}
	}
}

Eigen::VectorXi CB::rwm(size_t t, size_t l) {
	
	int total = N(l);
	int remaining = total;

	Eigen::VectorXi battles(k); //will hold number of soldiers to put into each battle

	auto f = get_partition(t, l);
	
	// if first round
	if ( (hist_loss(l).array() == 0).all() ) {
		//std::cout << "Using random";
		battles = rand_comp_n_k(total, k);
	}

	else {
		auto f = get_partition(t, l);
		//for each battle
		for (size_t h = k - 1; h > 0; --h) {

			Vecld weights = Vecld::Zero(total + 1); //will hold probability of allocating y soldiers to battle h for y = 0, ..., N[l]
			auto beta_vec = Eigen::ArrayXd::Constant(remaining + 1, beta);
			auto y = Eigen::ArrayXi::LinSpaced(remaining + 1, 0, remaining);
			weights(Eigen::seq(0, remaining)) = pow(beta, hist_loss(l).row(h)(Eigen::seq(0, remaining))) * f(h - 1, Eigen::seq(remaining, 0, Eigen::fix<-1>)) / f(h, remaining);

			if (std::abs(weights.sum() - 1) > 1e-5) {
				std::cout << "Time: " << t << '\n' << "Sum of probabilites: " << weights.sum() << '\n';
				throw(std::runtime_error("Probabilities do not sum to 1"));
			}

			//sample from discrete distribution
			//std::random_device rd;  // Will be used to obtain a seed for the random number engine
			//std::mt19937 gen(rd());// ^ (static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()))); // Standard mersenne_twister_engine seeded with rd()
			//std::cout << rd() << '\n';
			std::discrete_distribution<int> dist(weights.data(), weights.data() + weights.size());

			battles(h) = dist(gen);
			remaining -= battles(h);
		}
		battles(0) = remaining;
	}

	
	return battles;
}

Arrld CB::get_partition(size_t t, size_t l) {
	int total = N(l);
	Arrld f = Arrld::Zero(k, total + 1);
	Eigen::ArrayXd beta_vec = Eigen::ArrayXd::Constant(total + 1, beta);

	//base case
	f.row(0) = pow(beta, hist_loss(l).row(0));

	//otherwise
	for (size_t h = 1; h < k; ++h) {
		for (size_t y = 0; y <= total; ++y) {
			f(h, y) = (pow(beta, hist_loss(l).row(h)(Eigen::seq(0, y))) * f(h - 1, Eigen::seq(y, 0, Eigen::fix<-1>))).sum();
		}
	}
	return f;
}


Arrld CB::get_loss(size_t t, size_t l) {
	auto values = W.replicate(1, N(l) + 1).array();
	Arrld loss = Arrld::Zero(k, N(l) + 1);
	if (lt == loss_type::zero_one){
		auto m_greater = strategies(1 - l).row(t).replicate(N(l) + 1, 1).transpose() > Eigen::ArrayXi::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose();
		auto casted_m_greater = m_greater.cast<double>();
		auto m_equal = strategies(1 - l).row(t).replicate(N(l) + 1, 1).transpose() == Eigen::ArrayXi::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose();
		auto casted_m_equal = m_equal.cast<double>();
		loss = (values * casted_m_greater + ((double)1 / L) * values * casted_m_equal).cast<long double>();

	}

	else if (lt == loss_type::popular_vote) {
		Eigen::ArrayXXd m1 = strategies(1 - l).row(t).replicate(N(l) + 1, 1).transpose().cast<double>();
		Eigen::ArrayXXd m2 = Eigen::ArrayXi::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose().cast<double>();
		Eigen::ArrayXXd m3 = m1 + m2;
		loss = (values * m1.binaryExpr(m3, [](auto x, auto y) { return y == 0 ? 0.5 : x / y; })).cast<long double>();
		
	}

	else if (lt == loss_type::electoral_vote) {
		// loss is the probability that the opponent wins with their allocation
		for (size_t h = 0; h < k; ++h) {
			for (size_t x = 0; x <= N(l); ++x) {
				int proportionality = 10; //must be even
				int nh = W(h) * sum_of_values * proportionality; // proportional to the value of each battle 

				//P(X <= k) = 1 - incbeta(k+1, n-k), we want P(X >= k) = 1 - P(X <= k-1) = 1 - bincdf(k-1, n, p) = incbeta(k, n-(k-1)) 
				//k = (n / 2) + 1 for player 0 (player 1 must win majority) k = (n / 2) for player 1 (player 0 wins in tie)
				//p = opponent allocation / (opponent allocation + your proposed allocation)
				long double p;
				
				if (strategies(1 - l)(t, h) == 0 && x == 0) {
					//p = 0;
					loss(h, x) = W(h) * 0.5;
					//loss(h, x) = 1;
				}
				else {
					p = 1.* x / ((double)strategies(1 - l)(t, h) + (double)x);
					int kappa = (nh / 2) - 1;

					/*
					if (l == 1) {
						kappa = (nh / 2);

					}
					else {
						kappa = (nh / 2) + 1;
					}
					loss(h, x) = W(h) * incbeta(kappa, nh - (kappa - 1), p); // + half the probability of getting exactly ui/2 voters*/
					double cum_pr_half_minus_one = 1 - incbeta((double)kappa + 1, nh - (double)kappa, p);
					double pr_half = (1 - incbeta((double)kappa + 2, nh - ((double)kappa + 1), p)) - cum_pr_half_minus_one;
			
					loss(h, x) = W(h) * (cum_pr_half_minus_one + 0.5 * pr_half);
				}
			}
		}
	}
	
	else if (lt == loss_type::ev_adj) {
		
		double delta[] = { 0.8, 0.3, 0, 0, -0.3, -0.8 }; //advantage for player 1.
		//double delta[] = { 0.1, 0.05, 0, 0, -0.05, -0.1 };
		//double delta[] = { -0.08, -0.04, -0.02, 0.01, 0.06, 0 }; //NC, GA, AZ, MI, WI, PA
		for (size_t h = 0; h < k; ++h) {
			for (size_t x = 0; x <= N(l); ++x) {
				int proportionality = 10; //must be even
				int nh = W(h) * sum_of_values * proportionality; // proportional to the value of each battle 

				//P(X <= k) = 1 - incbeta(k+1, n-k), we want P(X >= k) = 1 - P(X <= k-1) = 1 - bincdf(k-1, n, p) = incbeta(k, n-(k-1)) 
				//k = (n / 2) + 1 for player 0 (player 1 must win majority) k = (n / 2) for player 1 (player 0 wins in tie)
				//p = opponent allocation / (opponent allocation + your proposed allocation)
				long double p;
				

				if (strategies(1 - l)(t, h) == 0 && x == 0) {
					//p = 0;
					loss(h, x) = W(h) * 0.5;
					//loss(h, x) = 1;
				}
				else {
					p = 1. * x / (strategies(1 - l)(t, h) + x);
					int kappa;
					if (l == 0) {
						double required = (nh / 2.) - (delta[h] * nh / 2.) - 1. + 0.5;
						kappa = (int)required;
					}
					else {
						double required = (nh / 2.) + (delta[h] * nh / 2.) - 1. + 0.5;
						kappa = (int)required;
					}
					
					double cum_pr_half_minus_one = 1 - incbeta(kappa + 1, nh - kappa, p);
					double pr_half = (1 - incbeta(kappa + 2, kappa + 1, p)) - cum_pr_half_minus_one;
					loss(h, x) = W(h) * (cum_pr_half_minus_one + 0.5 * pr_half);
				}
			}
		}
	}

	return loss;
}


Arrld CB::get_reward(size_t t, size_t l) {
	auto values = W.replicate(1, N(l) + 1).array();
	Arrld reward = Arrld::Zero(k, N(l) + 1);
	if (lt == loss_type::zero_one) {
		auto m_less = strategies(1 - l).row(t).replicate(N(l) + 1, 1).transpose() < Eigen::ArrayXi::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose();
		auto casted_m_less = m_less.cast<double>();
		auto m_equal = strategies(1 - l).row(t).replicate(N(l) + 1, 1).transpose() == Eigen::ArrayXi::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose();
		auto casted_m_equal = m_equal.cast<double>();
		reward = (values * casted_m_less + ((double)1 / L) * values * casted_m_equal).cast<long double>();

	}

	else if (lt == loss_type::popular_vote) {
		Eigen::ArrayXXd m1 = Eigen::ArrayXi::LinSpaced(N(l) + 1, 0, N(l)).rowwise().replicate(k).transpose().cast<double>(); 
		Eigen::ArrayXXd m2 = strategies(1 - l).row(t).replicate(N(l) + 1, 1).transpose().cast<double>();
		Eigen::ArrayXXd m3 = m1 + m2;
		reward = (values * m1.binaryExpr(m3, [](auto x, auto y) { return y == 0 ? 0.5 : x / y; })).cast<long double>();

	}

	else if (lt == loss_type::electoral_vote) {
		// loss is the probability that the opponent wins with their allocation
		for (size_t h = 0; h < k; ++h) {
			for (size_t x = 0; x <= N(l); ++x) {
				int proportionality = 10; //must be even
				int nh = W(h) * sum_of_values * proportionality; // proportional to the value of each battle 

				//P(X <= k) = 1 - incbeta(k+1, n-k), we want P(X >= k) = 1 - P(X <= k-1) = 1 - bincdf(k-1, n, p) = incbeta(k, n-(k-1)) 
				//k = (n / 2) + 1 for player 0 (player 1 must win majority) k = (n / 2) for player 1 (player 0 wins in tie)
				//p = opponent allocation / (opponent allocation + your proposed allocation)
				long double p;

				if (strategies(1 - l)(t, h) == 0 && x == 0) {
					//p = 0;
					reward(h, x) = W(h) * 0.5;
					//loss(h, x) = 1;
				}
				else {
					p = 1. * x / (strategies(1 - l)(t, h) + x);
					int kappa = (nh / 2) - 1;

					double cum_pr_half_minus_one = 1 - incbeta(kappa + 1, nh - kappa, p);
					double pr_half = (1 - incbeta(kappa + 2, nh - (kappa + 1), p)) - cum_pr_half_minus_one;
					reward(h, x) = W(h) * ( (1 - cum_pr_half_minus_one) + 0.5 * (1 - pr_half));
				}
			}
		}
	}

	else if (lt == loss_type::ev_adj) {
		
		double delta[] = { 0.8, 0.3, 0, 0, -0.3, -0.8 }; //advantage for player 1.
		//double delta[] = { 0.1, 0.05, 0, 0, -0.05, -0.1 };
		//double delta[] = { -0.08, -0.04, -0.02, 0.01, 0.06, 0 }; //NC, GA, AZ, MI, WI, PA
		for (size_t h = 0; h < k; ++h) {
			for (size_t x = 0; x <= N(l); ++x) {
				int proportionality = 10; //must be even
				int nh = W(h) * sum_of_values * proportionality; // proportional to the value of each battle 

				//P(X <= k) = 1 - incbeta(k+1, n-k), we want P(X >= k) = 1 - P(X <= k-1) = 1 - bincdf(k-1, n, p) = incbeta(k, n-(k-1)) 
				//k = (n / 2) + 1 for player 0 (player 1 must win majority) k = (n / 2) for player 1 (player 0 wins in tie)
				//p = opponent allocation / (opponent allocation + your proposed allocation)
				long double p;


				if (strategies(1 - l)(t, h) == 0 && x == 0) {
					//p = 0;
					reward(h, x) = W(h) * 0.5;
					//loss(h, x) = 1;
				}
				else {
					p = 1. * x / (strategies(1 - l)(t, h) + x);
					int kappa;
					if (l == 0) {
						double required = (nh / 2.) - (delta[h] * nh / 2.) - 1. + 0.5;
						kappa = (int)required;

					}
					else {
						double required = (nh / 2.) + (delta[h] * nh / 2.) - 1. + 0.5;
						kappa = (int)required;

					}
					double cum_pr_half_minus_one = 1 - incbeta(kappa + 1, nh - kappa, p);
					double pr_half = (1 - incbeta(kappa + 2, kappa + 1, p)) - cum_pr_half_minus_one;
					reward(h, x) = W(h) * ( (1 - cum_pr_half_minus_one) + 0.5 * (1 - pr_half));
				}
			}
		}
	}

	return reward;
}

//Implements Algorithm 2 from "Efficient Computation of Approximate Equilibria in Discrete Colonel Blotto Games, Vu et.al
long double CB::get_best_hist_loss(size_t time, size_t l, Arrld mat) {

	//pi(j, i) represents the optimal cumulative loss for player l when they are allowed to use 
	//j troops over the first i battlefields
	Arrld pi = Arrld(N(l) + 1, k);
	Arrld H = mat;
	
	// for each possible amount of troops
	for (size_t j = 0; j <= N(l); ++j) {
		//pi(j, 0) = 0;
		// for each battlefield
		pi(j, 0) = H(Eigen::seq(0, j), 0).minCoeff();

		for (size_t i = 1; i < k; ++i) {
			long double min = (pi(Eigen::seq(0, j), i - 1) + H(Eigen::seq(j, 0, Eigen::fix<-1>), i)).minCoeff();
			pi(j, i) = min;
		}

	}

	/*
	int j = N(l);

	std::vector<int> br(k);
	// loop through battlefields going backwards
	for (int i = k - 1; i >= 0; --i) {
		long double min = DBL_MAX;
		int argmin = -1;

		if (i > 0) {
			for (int t = 0; t <= j; ++t) {
				if (pi(j - t, i - 1) + H(t, i) < min) {
					min = pi(j - t, i - 1) + H(t, i);
					argmin = t;
				}
			}
			br[i] = argmin;
			j -= br[i];
		}
		else {
			br[i] = j;
		}
	}
	long double s = 0;
	

	if (l == 0) {
		std::cout << "best response: ";
	}
	for (size_t h = 0; h < k; ++h) {
		s += mat(br[h], h);
		if (l == 0) {

			std::cout << br[h] << ',';
		}
	}	
	*/

	return pi(N(l), k - 1);
}

void CB::run_test(std::string prefix, std::string suffix) {

	/*
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
		//w = electoral_weights;
		w = Eigen::VectorXd::Constant(battles, 1);
	}

*/
//double beta = 0.95;

	auto begin = std::chrono::high_resolution_clock::now();
	int it = this->run();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin);

	auto reg = this->get_regrets();

	std::string file_name = prefix + "results_k-" + std::to_string(k) + "_n-" + std::to_string(N[0]) +
		"_optimistic-" + std::to_string(optimistic) + "_init-" + std::to_string(init) + "_loss-" + std::to_string(lt) + "_I0-"  + std::to_string(init_factor) + "_" + suffix +  ".txt";


	thread_local std::ofstream f;
	f.open(file_name);

	f << "Battles: " << std::to_string(k) << '\n';

	f << "Soldiers: [" << std::to_string(N[0]) << ", " << std::to_string(N[1]) << ']' << '\n';

	f << "Values: ";
	for (size_t i = 0; i < k; ++i) {
		f << std::to_string(this->get_sum() * 1. * W[i]) << " ";
	}
	
	f  << '\n';

	long double avg_reg = -1;
	long double min_reg = 100;
	size_t a_min;
	for (size_t i = 0; i < reg.size(); ++i) {
		if (reg[i][3] < min_reg && reg[i][3] != 0) {
			min_reg = reg[i][3];
			a_min = i;
		}
	}

	if (it == TMAX) {
		f << "WARNING: MAX ITERATIONS EXECUTED" << '\n';
		avg_reg = this->get_regrets().back().back();
	}

	else {
		avg_reg = min_reg;
	}

	f << "Iterations: " << it << '\n';

	
	f << "Final regret: " << std::to_string(avg_reg) << '\n';

	int total_seconds = elapsed.count();
	int hours = total_seconds / 3600;
	total_seconds -= hours * 3600;

	int minutes = total_seconds / 60;
	total_seconds -= minutes * 60;

	f << "Runtime: " << std::to_string(hours) << "h "
		<< std::to_string(minutes) << "m "
		<< std::to_string(total_seconds) << "s " << '\n' << '\n';

	f << "Last 3 rounds: " << '\n';

	auto strategies = this->get_strategies();

	if (it == 0) {
		for (size_t j = 0; j < L; ++j) {
			for (size_t h = 0; h < k; ++h) {
				f << strategies(j)(it, h) << " ";
			}
			f << '\n';
		}
		f << '\n' << '\n';
	}

	else {
		for (size_t i = it - 2; i <= it; ++i) {
			f << "t = " << i << ": " << '\n';
			for (size_t j = 0; j < L; ++j) {
				for (size_t h = 0; h < k; ++h) {
					f << strategies(j)(i, h) << " ";
				}
				f << '\n';
			}
			f << '\n' << '\n';
		}
	}
	
	f << "Player 1 distribution: " << '\n';
	f << this->get_weights() << '\n' << '\n';

	/*
	f << "Player 1 Average Allocation: " << '\n';
	f << this->get_avg_al();
	*/

	

	//auto min_reg = *std::min_element(reg.begin() + 1, reg.end());
	//auto a_min = std::distance(reg.begin() + 1, std::min_element(reg.begin() + 1, reg.end()));
	Eigen::VectorXd best_avg_strat = Eigen::VectorXd::Zero(k);
	Eigen::VectorXd best_avg_strat_1 = Eigen::VectorXd::Zero(k);
	int stop_time = (a_min) * (int)T0;
	
	//int stop_time = (int)TMAX;
	//std::cout << stop_time << '\n';
	//std::cout << strategies(0)(stop_time, 0);

	for (int i = 0; i < k; ++i) {
		for (int j = 0; j <=  stop_time; ++j) {
			best_avg_strat(i) += (double) strategies(0)(j, i);
			best_avg_strat_1(i) += (double)strategies(1)(j, i);
		}
		best_avg_strat(i) /= ((double)stop_time + 1);
		best_avg_strat_1(i) /= ((double)stop_time + 1);
	}

	f << "Average Allocations: " << '\n';
	f << best_avg_strat.transpose() << '\n' << best_avg_strat_1.transpose() << '\n';
	f << '\n' <<  '\n' << "Best regret:" << '\n' << std::to_string(min_reg) << '\n' << '\n';
	f << "Iteration achieved" << '\n' << std::to_string((int)T0 * (a_min));

	f.close();

	//std::filesystem::create_directory("regrets");
	std::ofstream f1;
	std::string file_name_1 = prefix + "regrets_k-" + std::to_string(k) + "_n-" + std::to_string(N[0]) +
		"_optimistic-" + std::to_string(optimistic) + "_init-" + std::to_string(init) + "_loss-" + std::to_string(lt) + "_I0-" + std::to_string(init_factor) + "_" + suffix + ".txt";

	f1.open(file_name_1);

	/*
	std::string file_name_2 = prefix + "eq_k_" + std::to_string(k) + "_n_" + std::to_string(N[0]) +
		"_opt_" + std::to_string(optimistic) + "_init_" + std::to_string(init) + "_loss_" + std::to_string(lt) + "_I0_" + std::to_string(init_factor) + "_" + suffix + ".txt";

	std::ofstream f2;
	f2.open(file_name_2);
	*/
	
	
	f1 << "Time,R1,R2,TR" << '\n';
	//f2 << "Time,D1,D2,MD" << '\n';
	for (size_t i = 0; i < reg.size(); ++i) {
		f1 << std::to_string(reg[i][0]) << "," << std::to_string(reg[i][1]) << "," << std::to_string(reg[i][2]) << "," << std::to_string(reg[i][3]) << "," << '\n';
		//f2 << std::to_string(eq_dis[i][0]) << "," << std::to_string(eq_dis[i][1]) << "," << std::to_string(eq_dis[i][2]) << "," << std::to_string(eq_dis[i][3]) << "," << '\n';


	}
	f1.close();
	//f2.close();
}

/*
extern "C" {
	void CB_test_run(size_t T, size_t L, size_t k, int N[], double W[], double beta, size_t T0, double tol, bool optimistic,
						CB::init_type init, size_t init_factor, CB::loss_type lt) {
		CB test = CB(T, L, k, N, W, beta, T0, tol, optimistic, init, init_factor, lt);
		std::string prefix = "";
		std::string suffix = "";
		test.run_test(prefix, suffix);
	}
	//CB* CB_new(size_t T, size_t L, size_t k, int N[], double W[], double beta, size_t T0, double tol, bool optimistic) { return new CB(T, L, k, N, W, beta, T0, tol, optimistic); }
	//int CB_run(CB* game) { return game->run(); }
	/*
	std::vector<std::vector<std::vector<int>>> CB_get_strategies(CB* game) { return game->get_strategies(); }
	std::vector<double> CB_get_regrets(CB* game) { return game->get_regrets(); }
	size_t CB_get_TMAX(CB* game) { return game->get_tmax(); }
	
}
*/
