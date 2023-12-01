#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1418772@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o final_project-%j
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

python prompt.py --cache_dir ${CACHE_DIR} --prompt zero_shot --val_size 76  
python prompt.py --cache_dir ${CACHE_DIR} --prompt one_shot --val_size 76  
python prompt.py --cache_dir ${CACHE_DIR} --prompt two_shot --val_size 76  

python gradient_based.py --cache_dir ${CACHE_DIR} --analysis_file val.jsonl

