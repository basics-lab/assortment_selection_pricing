#!/bin/bash
#SBATCH --job-name=synt_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --error=slurm_outputs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

for i in {1..10}
do
  /Users/erginbas/opt/anaconda3/bin/python3 /Users/erginbas/Documents/Research\ Codes/assortment_selection_pricing/experiments.py &
done
