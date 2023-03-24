#!/bin/bash
#SBATCH --job-name=dino          # create a short name for your job
#SBATCH --output="default_params_epochs=5-%j.out"      # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node

source /opt/conda/bin/activate
conda activate pytorch

python3 train_dino.py

conda deactivate