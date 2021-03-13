#!/bin/bash -l

#SBATCH --job-name=maa_sepsis_src       # Job name
#SBATCH --nodes=1                       # Run all processes on a single node	
#SBATCH --ntasks-per-node=1             # Run a single task		
#SBATCH --cpus-per-task=40              # Number of CPU cores per task
#SBATCH --gres=gpu:0                    # Number of GPU per task in the node
#SBATCH --mem=256GB                     # Job memory request
#SBATCH --output=XGBmodel_training.log  # Standard output and error log

# Setup
export NCCL_DEBUG=INFO 
export PYTHONFAULTHANDLER=1
pythonExec=$1 

echo "Running job in slurm"

cd $HOME/projects/SepsisProject

date
srun $pythonExec XGBmodel.py
date 

echo "Terminated job in slurm"