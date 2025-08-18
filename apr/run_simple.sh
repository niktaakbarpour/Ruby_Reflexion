#!/bin/bash
#SBATCH --job-name=simple_pass_at1_iter11_Qwen_40
#SBATCH --time=111:00:00
#SBATCH --account=st-fhendija-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/simple_pass_at1_iter11_Qwen_40_output.txt
#SBATCH --error=logs/simple_pass_at1_iter11_Qwen_40_error.txt

cd $SLURM_SUBMIT_DIR

# Load CUDA
#module load intel-oneapi-compilers/2023.1.0
module load gcc
module load cuda
module load miniconda3

# Set CUDA_HOME if necessary
#export CUDA_HOME=/usr/local/cuda-11.3  # Replace with your actual CUDA path

# Activate virtual environment
source activate /home/mahdiehs/miniconda3/envs/ruby_env_conda

export HF_HOME=/scratch/st-fhendija-1/mahdiehs/cache
export TRITON_CACHE_DIR=/scratch/st-fhendija-1/mahdiehs/.triton

python main.py \
--run_name "simple_pass_at1_iter11_Qwen_40" \
--root_dir "root" \
--dataset_path ./benchmarks/compact_ruby.jsonl \
--strategy "simple" \
--language "rb" \
--model "Qwen/Qwen2.5-Coder-7B-Instruct" \
--model_path "/scratch/st-fhendija-1/mahdiehs/Projects/models/Qwen2.5-Coder-7B-Instruct" \
--pass_at_k "1" \
--max_iters "11" \
--verbose | tee ./logs/simple_Qwen_40