#!/bin/bash

## DON'T USE SPACES AFTER COMMAS

# You must specify a valid email address!
#SBATCH --mail-user=negin.ghamsarian@unibe.ch
# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=FAIL,END


# Job name
#SBATCH --job-name="ENCORE_RETOUCH_DA"
# Partition
#SBATCH --partition=gpu-invest # all, gpu, phi, long

# Runtime and memory
#SBATCH --time=24:00:00    # days-HH:MM:SS
#SBATCH --mem-per-cpu=4G # it's memory PER CPU, NOT TOTAL RAM! maximum RAM is 246G in total
# total RAM is mem-per-cpu * cpus-per-task

# maximum cores is 20 on all, 10 on long, 24 on gpu, 64 on phi!
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1

# on gpu partition
#SBATCH --gres=gpu:rtx3090:1

#SBATCH --output=../DA_MRI_MICCAI23_Results/code_outputs/ENCORE_AugLoss_UnsupOnTrain_%j.out_
#SBATCH --error=../DA_MRI_MICCAI23_Results/code_errors/ENCORE_AugLoss_UnsupOnTrain_%j.err

module load Python/3.9.5-GCCcore-10.3.0 #cuda/10.2.89 cuDNN/8.2.1.32-CUDA-11.3.1

#eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
#conda activate /storage/homefs/ng22l920/anaconda3/envs/PyTorch_GPU

#source /storage/homefs/ng22l920/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh

#eval "$(conda shell.bash hook)"
#source activate ~//anaconda3/etc/profile.d/conda.sh
#source /storage/homefs/ng22l920/anaconda3/etc/profile.d/conda.sh
#conda init powershell 
#conda activate ~//anaconda3/envs/PyTorch_GPU
##conda init --all --dry-run --verbose
#source activate /storage/homefs/ng22l920/anaconda3/envs/PyTorch_GPU
#python Test_VisualStudio.py 
srun /storage/homefs/ng22l920/anaconda3/envs/PyTorch_RTX3090/bin/python3 ENCORE_AugLoss.py --config 'configs_MRI2_scSENet.Config_ENCORE_AugLoss_UnsupOnTrain_T85'