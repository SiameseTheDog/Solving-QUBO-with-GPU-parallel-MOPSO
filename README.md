# ISYE6679-Project
## Intruduction
This is paper replication of:

Noriyuki Fujimoto and Kouki Nanai. 2021. Solving QUBO with GPU parallel MOPSO. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '21). Association for Computing Machinery, New York, NY, USA, 1788–1794. https://doi.org/10.1145/3449726.3463208



## Files:
- single_threaded_CPU.cpp - implementation of sequential programming without any parallelization


- CPUprogramming.cpp - implementation of both single thread and multiple threads using CBLAS with timing. Error is calculated on results from multi-threaded way.
  To complie, using command 
  ```
  g++ -o CPUprogramming CPUprogramming.cpp -lopenblas -O3
  ```
  To run, use command
  ```
  ./CPUprogramming
  ```

- mopso.cu
  To complie, using command with Makefile
  ```
  make
  ```
  To run, use command
  ```
  ./mopso <TestCaseFileName>
  ```
  The test case matrix shoule be in .txt format. The first line in txt file should be the number of rows and starting from the second line, it should list the matrix numbers separated by spaces.
