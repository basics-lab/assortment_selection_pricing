#!/bin/bash
#SBATCH --job-name=menu_price
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --error=slurm_outputs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

parallel --progress -j 10 python3 experiments.py {1} {2} {3} {4} {5} {6} ::: \
5 ::: `#d` \
0.3 ::: `#L0` \
2000 ::: `#T` \
5 ::: `#N`\
5 ::: `#K` \
100 ::: `#T0_low`
{1..5}