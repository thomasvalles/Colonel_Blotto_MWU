#include "CB.h"
#include "Part.h"



CB::CB(size_t _T, size_t _L, size_t _k, Eigen::VectorXi _N, Eigen::VectorXd _W, double _beta, size_t _T0, double _tol, bool _optimistic) :
	TMAX(_T), L(_L), k(_k), N(_N), W(_W), beta(_beta), T0(_T0), tol(_tol), optimistic(_optimistic) {

	std::srand((unsigned int)time(0));

	//(player)(time, battlefield)
	strategies = Eigen::Vector<Eigen::ArrayXXi, Eigen::Dynamic>(L);

	//(player)(battlefield, amount)
	hist_loss = Eigen::Vector<Arrld, Eigen::Dynamic>(L);

	for (size_t i = 0; i < L; ++i) {
		strategies(i) = Eigen::ArrayXXi::Zero(TMAX, k);
		hist_loss(i) = Arrld::Zero(k, N(i) + 1);
	}
	W.normalize();
}

int CB::run() {
	int i = 0;
	do {
		//for each player
		for (size_t j = 0; j < L; ++j) {
			strategies(j).row(i) = rwm(i, j);
		}
		update_hist_loss(i);

		//every T0 rounds 
		if ((i) % T0 == 0) {
			//std::cout << i << '\n';
			long double regret = 0;

			//for each player
			for (size_t l = 0; l < L; ++l) {

				long double best_hist_loss = get_best_hist_loss(i, l); //get the best hist loss
				//regret += hist_loss(l)(Eigen::seq(0, k-1), strategies(l)(i, Eigen::seq(0, k-1))).sum();
				
				for (size_t h = 0; h < k; ++h) {
					regret += hist_loss(l)(h, strategies(l)(i, h)); //get the loss for the chosen strategy 
				}
				
				regret -= best_hist_loss; //subtract the best 
		
			}
			regrets.push_back(regret / (double(i) + 1)); //take the average
			if (isnan(regrets.back())) {
				std::cout << "isnan. regret:  " << regret << " time: " << std::to_string(double(i) + 1);
			}
		}
		++i;
	} while ((i < TMAX) && (regrets.back() > tol));

	return (i - 1);
}

void CB::update_hist_loss(size_t time) {
	//for each player
	for (size_t l = 0; l < L; ++l) {
		if (optimistic) {
			if (time > 0) {
				hist_loss(l) += 2 * get_loss(time, l) - get_loss(time - 1, l);
			}
			else {
				hist_loss(l) += 2 * get_loss(time, l);
			}
		}
		else {
			hist_loss(l) += get_loss(time, l);
		}
	}
}

Eigen::VectorXi CB::rwm(size_t t, size_t l) {

	int total = N(l);
	int remaining = total;

	Eigen::VectorXi battles(k); //will hold number of soldiers to put into each battle

	// if first round
	if (t == 0) {
		battles = rand_comp_n_k(total, k); //get random composition
	}

	else {
		auto f = get_partition(t, l);

		//for each battle
		for (size_t h = k - 1; h > 0; --h) {
			
			Vecld weights = Vecld::Zero(total + 1); //will hold probability of allocating y soldiers to battle h for y = 0, ..., N[l]
			auto beta_vec = Eigen::ArrayXd::Constant(remaining + 1, beta);
			auto y = Eigen::ArrayXi::LinSpaced(remaining, 0, remaining);
			weights(Eigen::seq(0, remaining)) = pow(beta, hist_loss(l).row(h)(Eigen::seq(0, remaining))) * f(h - 1, Eigen::seq(remaining, 0, Eigen::fix<-1>)) / f(h, remaining);

			if (std::abs(weights.sum() - 1) > 1e-5) {
				std::cout << "Time: " << t << '\n' << "Sum of probabilites: " << weights.sum() << '\n';
				throw(std::runtime_error("Probabilities do not sum to 1"));
			}

			//sample from discrete distribution
			std::random_device rd;  // Will be used to obtain a seed for the random number engine
			std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
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
		
		/*
		Eigen::Tensor<long double, 1> beta_vec(total+1);
		Eigen::Tensor<long double, 1> f_vec(total+1);
		for (size_t y = 0; y <= total; ++y) {
			beta_vec(y) = std::pow(beta, hist_loss(l)(h, y));
			f_vec(y) = f(h - 1, y);
		}
		auto x = beta_vec.convolve(f_vec, 0);
		//std::cout <<beta_vec.convolve(f_vec, 0);
		*/
		
		for (size_t y = 0; y <= total; ++y) {
			f(h, y) = (pow(beta, hist_loss(l).row(h)(Eigen::seq(0, y))) * f(h - 1, Eigen::seq(y, 0, Eigen::fix<-1>))).sum();
			/*
			long double s = 0;
			for (size_t x = 0; x <= y; ++x) {
				s += std::pow(beta, hist_loss(l)(h, x)) * f(h - 1, y - x);
			}
			f(h, y) = s;
			*/
		}
	}
	return f;
}


Arrld CB::get_loss(size_t t, size_t l) {
	auto m_greater = strategies(1-l).row(t).replicate(N(l), 1).transpose() > Eigen::ArrayXi::LinSpaced(N(l), 0, N(l)).rowwise().replicate(k).transpose();
	auto casted_m_greater = m_greater.cast<double>();
	auto m_equal = strategies(1 - l).row(t).replicate(N(l), 1).transpose() == Eigen::ArrayXi::LinSpaced(N(l), 0, N(l)).rowwise().replicate(k).transpose();
	auto casted_m_equal = m_equal.cast<double>();

	auto weights = W.replicate(1, N(l)).array();
	
	Arrld loss = (weights * casted_m_greater + ((double) 1 / L) * weights * casted_m_equal).cast<long double>();

	return loss;
}

//Implements Algorithm 2 from "Efficient Computation of Approximate Equilibria in Discrete Colonel Blotto Games, Vu et.al
long double CB::get_best_hist_loss(size_t time, size_t l) {

	//pi(j, i) represents the optimal cumulative loss for player l when they are allowed to use 
	//j troops over the first i battlefields
	Arrld pi = Arrld(N(l) + 1, k);
	Arrld H = Arrld(N(l) + 1, k);
	if (optimistic) {
		H = (hist_loss(l) - get_loss(time, l)).transpose();
	}
	else {
		H = hist_loss(l).transpose();
	}


	// for each possible amount of troops
	for (size_t j = 0; j <= N(l); ++j) {
		//pi(j, 0) = 0;
		// for each battlefield
		pi(j, 0) = H(Eigen::seq(0, j), 0).minCoeff();
		//pi(j, Eigen::seq(1, k - 1)) = (pi(Eigen::seq(0, j), Eigen::seq(0, k - 2)) + H(Eigen::seq(j, 0, Eigen::fix<-1>), Eigen::seq(0, k - 1))).colwise().minCoeff();
		
		for (size_t i = 1; i < k; ++i) {
			long double min = (pi(Eigen::seq(0, j), i - 1) + H(Eigen::seq(j, 0, Eigen::fix<-1>), i)).minCoeff();	
			pi(j, i) = min;
		}
		
	}

	// UNCOMMENT IF YOU WANT THE ACTUAL STRATEGY THAT ACHIEVES THE MIN LOSS, NOT THE MIN LOSS
	//std::cout << "Predicted min loss: " << pi[N[l]][k - 1] << '\n';
	/*
	int j = N[l];

	// loop through battlefields going backwards
	for (int i = k - 1; i >= 0; --i) {
		long double min = DBL_MAX;
		int argmin = -1;

		if (i > 0) {
			for (int t = 0; t <= j; ++t) {
				if (pi[j - t][i - 1] + H[t][i] < min) {
					min = pi[j - t][i - 1] + H[t][i];
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
	for (size_t h = 0; h < k; ++h) {
		s += hist_loss[l][h][br[h]];
	}
	//std::cout << "Actual min loss: " << s << '\n';
	*/
	return pi(N[l], k - 1);
}