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
5 10 ::: `#d` \
0.1 0.3 0.5 ::: `#L0` \
2000 ::: `#T` \
10 100 ::: `#N`\
5 10 ::: `#K` \
100 200 ::: `#T0_low`
{1..5}