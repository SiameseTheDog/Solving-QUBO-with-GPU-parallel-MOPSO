#!/bin/bash

# Request 16 CPU cores
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -o batch-%j.log

# Define the number of particles
numParticles=32768

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load the necessary modules
module load gcc openblas
module load hwloc  # Ensure hwloc is available on your system

# Define the output log file, including number of particles in the filename
outputLog="CPU_output_${numParticles}.log"

# Clear and then write CPU info directly into the main output log
echo "Collecting CPU Info for Job: $SLURM_JOB_ID" > $outputLog  # This overwrites existing content
echo "General CPU Info:" >> $outputLog
lscpu >> $outputLog
echo "SLURM Node Allocation Info:" >> $outputLog
scontrol show node $SLURM_NODELIST >> $outputLog

# Insert an empty line before starting the test cases
echo "" >> $outputLog

# Run the CPU executable for each test case and append outputs to the log file
# Test cases for group A
for i in {1..8}; do
    ./CPU "Test Cases/test_case_A$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog  # Empty line after group A

# Test cases for group B
for i in {1..10}; do
    ./CPU "Test Cases/test_case_B$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog  # Empty line after group B

# Continue for other groups, ensuring to echo an empty line after each group
# Test cases for group C
for i in {1..7}; do
    ./CPU "Test Cases/test_case_C$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog

# Test cases for group D
for i in {1..10}; do
    ./CPU "Test Cases/test_case_D$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog

# Test cases for group E
for i in {1..5}; do
    ./CPU "Test Cases/test_case_E$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog

# Test cases for group F
for i in {1..5}; do
    ./CPU "Test Cases/test_case_F$i.txt" $numParticles >> $outputLog
done
echo "" >> $outputLog
