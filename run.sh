#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 
#SBATCH --time=8:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=terry.cox@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_3-%j
#SBATCH --export=ALL

source /uufs/chpc.utah.edu/common/home/u1427155/miniconda3/etc/profile.d/conda.sh
conda activate train_plm

mkdir -p /scratch/general/vast/u1427155/huggingface_cache
export TRANSFORMER_CACHE="/scratch/general/vast/u1427155/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1427155/huggingface_cache"

ANALSIS_OUT=/uufs/chpc.utah.edu/common/home/u1427155/assignments/GradBased_Highlighting/out
FILE_IN=/uufs/chpc.utah.edu/common/home/u1427155/assignments/GradBased_Highlighting/data/failed.jsonl
MODEL_CHECKPOINT=/scratch/general/vast/u1427155/cs6966/assignment1/models/microsoft/deberta-v3-base-finetuned-imdb/checkpoint-12500
MODEL_OUT_DIR=/scratch/general/vast/u1427155/cs6966/assignment3/models
mkdir -p ${MODEL_OUT_DIR}
python src/assignment_4.py --analsis_dir ${ANALSIS_OUT} --model_checkpoint ${MODEL_CHECKPOINT} --a1_analysis_file ${FILE_IN} --num_labels "2" --output_dir ${ANALSIS_OUT}