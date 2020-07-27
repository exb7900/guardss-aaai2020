#!/bin/bash
#SBATCH -J run_def_util                # Job ID
#SBATCH -c 4                           # Number of cores
#SBATCH -N 1                           # Ensure that all cores are on one machine
#SBATCH -t 6-8:00                      # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe                       # Partition to submit to
#SBATCH --mem=4G                       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_%A_%a.out               # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_%A_%a.err               # File to which STDOUT will be written, %j inserts jobid
#SBATCH --array=1-24                   # runs 24 copies of the job

module load Anaconda3
source activate test_fp

mkdir -p $SLURM_ARRAY_TASK_ID
cd $SLURM_ARRAY_TASK_ID
cp ../U_dc.pkl .
cp ../U_du.pkl .
cp ../U_ac.pkl .
cp ../U_au.pkl .
cp ../loopList.pkl .
srun -c $SLURM_CPUS_PER_TASK python ../compare_defender_utilities_script.py 10 1 3 -nsh -g 1 -sr -np -lp -sg