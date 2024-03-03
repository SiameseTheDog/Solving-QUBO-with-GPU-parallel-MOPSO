# ISYE6679-Project
Paper replication of:

Noriyuki Fujimoto and Kouki Nanai. 2021. Solving QUBO with GPU parallel MOPSO. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '21). Association for Computing Machinery, New York, NY, USA, 1788–1794. https://doi.org/10.1145/3449726.3463208



Files:

single_threaded_CPU.cpp - implementation of sequential programming without any parallelization


CPUprogramming.cpp - implementation of single thread and multiple threads using CBLAS with timing.
To complie, run 
```
g++ -o CPUprogramming CPUprogramming.cpp -lopenblas -O3
```

