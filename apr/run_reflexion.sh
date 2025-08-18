#!/bin/bash
#SBATCH --job-name=reflexion_pass_at1_iter11_Deepseek
#SBATCH --time=111:00:00
#SBATCH --account=st-fhendija-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu_mem_32
#SBATCH --output=logs/reflexion_pass_at1_iter11_Deepseek_output.txt
#SBATCH --error=logs/reflexion_pass_at1_iter11_Deepseek_error.txt

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
--run_name "reflexion_pass_at1_iter11_Deepseek" \
--root_dir "root" \
--dataset_path ./benchmarks/c++/merged_cpp_validation.jsonl \
--strategy "reflexion" \
--language "cpp" \
--model "deepseek-ai/deepseek-coder-6.7b-instruct" \
--model_path "/scratch/st-fhendija-1/mahdiehs/Projects/models/deepseek-model" \
--pass_at_k "1" \
--max_iters "11" \
--verbose | tee ./logs/reflexion_Deepseek