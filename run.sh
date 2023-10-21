#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1418772@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_3-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CS6966

mkdir -p /scratch/general/vast/u1418772/huggingface_cache
export TRANSFORMER_CACHE="/scratch/general/vast/u1418772/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1418772/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1418772/cs6966/final_project/models
CACHE_DIR=/scratch/general/vast/u1418772/huggingface_cache
mkdir -p ${OUT_DIR}
mkdir -p ${CACHE_DIR}

python prompt.py --output_dir ${OUT_DIR} --cache_dir ${CACHE_DIR}

