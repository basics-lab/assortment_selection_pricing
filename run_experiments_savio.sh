#!/bin/bash
#SBATCH --job-name=menu_price
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --error=slurm_outputs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

parallel --ungroup -j 10 python3 experiments.py ::: `seq 10`