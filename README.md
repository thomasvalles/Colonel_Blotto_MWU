# Colonel_Blotto_MWU

1) Ensure that you have https://gitlab.com/libeigen/eigen cloned
2) To compile, use g++ -I/full/path/to/eigen/Eigen -std=c++17 CB.cpp Part.cpp Source.cpp 
3) Then run a.exe
4) Use CB.run_test(prefix, suffix) to run the algorithm using the specified parameters in Source.cpp. This function will make 2 files: one containing the regrets for each player, the other a log file containing the "optimal" allocation for the players and the time taken to run.

5) (Right now this code really only works for 2 players (L = 2))
