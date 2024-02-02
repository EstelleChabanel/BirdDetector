#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5120

module load CUDA
module load cuDNN

source activate pdm
python src/model/gridsearch.py