#!/bin/bash

#SBATCH -N 16                   # Request 16 nodes
#SBATCH --cpus-per-task=32      # Request 32 CPU cores per task
#SBATCH -o batch-%j.log
#SBATCH -t 1:00:00              
#SBATCH --gres=gpu:1             # Generic request for 1 GPU per node

# Define the number of particles
numParticles=32768																																	

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load the necessary modules (adjust module names as necessary for your system)
module load gcc
module load cuda   # Load a generic cuda module or equivalent for your system

# Define the output log file, including the number of particles in the filename
outputLog="result_16_32/GPU_output_${numParticles}.log"

# Insert an empty line before starting the test cases
echo "" >> $outputLog

# Run the GPU executable for each test case and append outputs to the log file
# Test cases for group A
for i in {1..8}; do
    ./mopso "../Test Cases/test_case_A$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog  # Empty line after group A

# Test cases for group B
for i in {1..10}; do
    ./mopso "../Test Cases/test_case_B$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog  # Empty line after group B

# Continue for other groups, ensuring to echo an empty line after each group
# Test cases for group C
for i in {1..7}; do
    ./mopso "../Test Cases/test_case_C$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog

# Test cases for group D
for i in {1..10}; do
    ./mopso "../Test Cases/test_case_D$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog

# Test cases for group E
for i in {1..5}; do
    ./mopso "../Test Cases/test_case_E$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog

# Test cases for group F
for i in {1..5}; do
    ./mopso "../Test Cases/test_case_F$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog
