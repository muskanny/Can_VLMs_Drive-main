#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition=ihub
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home2/muskan.singh/logs/smolvlm_aw_v2_%j.out
#SBATCH --job-name=smolvlm_v2

export HF_HOME=/home2/muskan.singh/hf_cache
export HF_HUB_DISABLE_XET=1

source /home2/muskan.singh/miniconda3/etc/profile.d/conda.sh
conda activate vlm

echo "Starting Mode A (with image)..."
python /home2/muskan.singh/code/eval.py     --model smolvlm     --category adverse_weather     --manifest /home2/muskan.singh/aw_manifest_v2.json     --suffix _v2

echo "Mode A done."

echo "Starting Mode C (no image)..."
python /home2/muskan.singh/code/eval.py     --model smolvlm     --category adverse_weather     --manifest /home2/muskan.singh/aw_manifest_v2.json     --suffix _v2     --no-image

echo "Mode C done."
