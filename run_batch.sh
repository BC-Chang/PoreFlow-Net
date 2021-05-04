#!/bin/bash

#SBATCH -J edist_pot
#SBATCH -o edist_pot.o%j
#SBATCH -e edist_pot.e%j
#SBATCH -p v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mail-user=bcchang@utexas.edu
#SBATCH --mail-type=all
#SBATCH -A EAR20009

module load conda
conda activate PorePot_Net

CUDA_VISIBLE_DEVICES=0 python ./train.py edist_detrendedpot2 e_pore e_stats &
#CUDA_VISIBLE_DEVICES=1 python ./train.py current_test3 &
#CUDA_VISIBLE_DEVICES=2 python ./train.py current_test3 &
#CUDA_VISIBLE_DEVICES=3 python ./train.py current_test3 &
wait
