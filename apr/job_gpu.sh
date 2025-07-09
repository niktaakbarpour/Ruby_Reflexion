#!/bin/bash
#SBATCH --job-name=task0
#SBATCH --time=00:30:00
#SBATCH --account=st-fhendija-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu_mem_16
#SBATCH --output=logs/output_embeddings.txt
#SBATCH --error=logs/error_embeddings.txt

cd $SLURM_SUBMIT_DIR

# Load CUDA
module load cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set CUDA_HOME if necessary
#export CUDA_HOME=/usr/local/cuda-11.3  # Replace with your actual CUDA path

# Activate virtual environment
source /scratch/st-fhendija-1/nikta/reflexion/programming_runs/.venv/bin/activate

export HF_HOME=/scratch/st-fhendija-1/nikta/huggingface_cache

python main.py \
  --run_name "reflexion_deepseek" \
  --root_dir "root" \
  --dataset_path ./benchmarks/tiny.jsonl \
  --strategy "reflexion" \
  --language "rb" \
  --model "deepseek-ai/deepseek-coder-6.7b-instruct" \
  --model_path "/scratch/st-fhendija-1/nikta/deep_model" \
  --pass_at_k "3" \
  --max_iters "3" \
  --verbose | tee ./logs/reflexion_deepseek
