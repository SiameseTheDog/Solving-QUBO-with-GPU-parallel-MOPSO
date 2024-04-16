# ISYE6679-Project
## Intruduction
This is paper replication of:

Noriyuki Fujimoto and Kouki Nanai. 2021. Solving QUBO with GPU parallel MOPSO. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '21). Association for Computing Machinery, New York, NY, USA, 1788–1794. https://doi.org/10.1145/3449726.3463208



## Files:


- CPUprogramming.cpp - implementation of both single thread and multiple threads using CBLAS with timing.
  This will store all info above in a file names "CPU_output_${numParticles}.log". Change the number of threads for openblas to switch between multi and single thread.
  
  To complie, using command 
  ```
  g++ CPUprogramming.cpp -lopenblas -O3 -march=native -funroll-loops -o CPU
  ```
  To run, use command
  ```
  ./CPU <matrixFileName> <numParticles>
  ```
  
- CPU.sbatch - submit all task cases and record running time, target values and CPU info. Remember to change number of particles.
  To run, using command
  ```
  sbatch cpu.sbatch
  ```

- CPU_result.zip - where all result logs are stored

  
- cuda/mopso.cu
  
  To complie, using command with Makefile
  ```
  make
  ```
  To run, use command
  ```
  ./mopso <TestCaseFileName> <numOfParticles>
  ```
  The test case matrix shoule be in .txt format. The first line in txt file should be the number of rows and starting from the second line, it should list the matrix numbers separated by spaces.
- cuda/CUDA.sbatch - submit jobs to slurm to find solutions for all matrics. Remember to change the value of numOfParticles

- cuda/result_for_val - results for one node and 4 cores per node.
