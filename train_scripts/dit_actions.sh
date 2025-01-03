#!/bin/bash
#SBATCH --partition=preemption
#SBATCH --qos=xlong
#SBATCH --job-name=DIT
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --mem=300G
#SBATCH --output=.slurm/DIT_actions.out.txt
#SBATCH --error=.slurm/DIT_actions.err.txt

module load CUDA/12.1.1
module load Python
source /scratch/igarcia945/venvs/transformers/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=32

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

accelerate launch --mixed_precision bf16 train_dit.py configs/train_dit_actions.yaml