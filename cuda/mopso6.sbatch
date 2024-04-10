#!/bin/bash

# Request 1 GPU core.
#SBATCH -N4 --gres=gpu:4

# Request 60 minutes of wall-clock time.
#SBATCH -t60

# Set the output log to be batch.log
#SBATCH -oReport-%j.log
#SBATCH -obatch.log

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load the cuda module so we can find the run-time libraries
module load cuda

# Run the code and redirect output to a temporary file
./mopso test_case_A1.txt >> MOPSO_output_16384.log
./mopso test_case_A2.txt >> MOPSO_output_16384.log
./mopso test_case_A3.txt >> MOPSO_output_16384.log
./mopso test_case_A4.txt >> MOPSO_output_16384.log
./mopso test_case_A5.txt >> MOPSO_output_16384.log
./mopso test_case_A6.txt >> MOPSO_output_16384.log
./mopso test_case_A7.txt >> MOPSO_output_16384.log
./mopso test_case_A8.txt >> MOPSO_output_16384.log
./mopso test_case_B1.txt >> MOPSO_output_16384.log
./mopso test_case_B2.txt >> MOPSO_output_16384.log
./mopso test_case_B3.txt >> MOPSO_output_16384.log
./mopso test_case_B4.txt >> MOPSO_output_16384.log
./mopso test_case_B5.txt >> MOPSO_output_16384.log
./mopso test_case_B6.txt >> MOPSO_output_16384.log
./mopso test_case_B7.txt >> MOPSO_output_16384.log
./mopso test_case_B8.txt >> MOPSO_output_16384.log
./mopso test_case_B9.txt >> MOPSO_output_16384.log
./mopso test_case_B10.txt >> MOPSO_output_16384.log
./mopso test_case_C1.txt >> MOPSO_output_16384.log
./mopso test_case_C2.txt >> MOPSO_output_16384.log
./mopso test_case_C3.txt >> MOPSO_output_16384.log
./mopso test_case_C4.txt >> MOPSO_output_16384.log
./mopso test_case_C5.txt >> MOPSO_output_16384.log
./mopso test_case_C6.txt >> MOPSO_output_16384.log
./mopso test_case_C7.txt >> MOPSO_output_16384.log
./mopso test_case_D1.txt >> MOPSO_output_16384.log
./mopso test_case_D2.txt >> MOPSO_output_16384.log
./mopso test_case_D3.txt >> MOPSO_output_16384.log
./mopso test_case_D4.txt >> MOPSO_output_16384.log
./mopso test_case_D5.txt >> MOPSO_output_16384.log
./mopso test_case_D6.txt >> MOPSO_output_16384.log
./mopso test_case_D7.txt >> MOPSO_output_16384.log
./mopso test_case_D8.txt >> MOPSO_output_16384.log
./mopso test_case_D9.txt >> MOPSO_output_16384.log
./mopso test_case_D10.txt >> MOPSO_output_16384.log
./mopso test_case_E1.txt >> MOPSO_output_16384.log
./mopso test_case_E2.txt >> MOPSO_output_16384.log
./mopso test_case_E3.txt >> MOPSO_output_16384.log
./mopso test_case_E4.txt >> MOPSO_output_16384.log
./mopso test_case_E5.txt >> MOPSO_output_16384.log
./mopso test_case_F1.txt >> MOPSO_output_16384.log
./mopso test_case_F2.txt >> MOPSO_output_16384.log
./mopso test_case_F3.txt >> MOPSO_output_16384.log
./mopso test_case_F4.txt >> MOPSO_output_16384.log
./mopso test_case_F5.txt >> MOPSO_output_16384.log

