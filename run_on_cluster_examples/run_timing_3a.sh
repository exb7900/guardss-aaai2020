#!/bin/bash
#SBATCH -J run_3a_time                 # Job ID
#SBATCH -c 4                           # Number of cores
#SBATCH -N 1                           # Ensure that all cores are on one machine
#SBATCH -t 6-8:00                      # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared                      # Partition to submit to
#SBATCH --mem=16G                      # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_%A_%a.out               # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_%A_%a.err               # File to which STDOUT will be written, %j inserts jobid
#SBATCH --array=1-24                   # runs 24 copies of the job 

module load Anaconda3
source activate test_fp

mkdir -p $SLURM_ARRAY_TASK_ID
cd $SLURM_ARRAY_TASK_ID
srun -c $SLURM_CPUS_PER_TASK python ../compare_heuristic_script.py 6 18 2 1 "all" -nsh -sp -np 
