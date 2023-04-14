# Colonel_Blotto_MWU

1) Ensure that you have https://gitlab.com/libeigen/eigen cloned
2) g++ -I/full/path/to/eigen/Eigen -std=c++17 CB.cpp Part.cpp Source.cpp -pthread
3) ./a.out
4) Source.cpp is currently set up to run 24 tests each on a separate thread:
  a) [10, 20, 30] battles, max_soldiers soldiers, optimistic/!optimistic, random_weights/!random_weights
  b) max_battles battles, [max_battles, 2 * max_battles, 3 * max_battles] soldiers, optimistic/!optimistic, random_weights/!random_weights
5) You can change max_soldiers and max_battles in lines 103 and 104 of Source.cpp.
6) The program will output the results of each test (including a description of the input, number of iterations executed, runtime, and 3 most recent strategies) to a file in a "results" folder.
