#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition=ihub
#SBATCH --nodelist=gnode095
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home2/muskan.singh/logs/internvl3_ji_v1_%j.out
#SBATCH --job-name=internvl3_ji

export HF_HOME=/home2/muskan.singh/hf_cache
export HF_HUB_DISABLE_XET=1

source /home2/muskan.singh/miniconda3/etc/profile.d/conda.sh
conda activate vlm

echo "Starting Mode A (with image)..."
python /home2/muskan.singh/code/eval.py     --model internvl3     --category junctions     --manifest /home2/muskan.singh/benchmark/junctions/ji_manifest_v1.json     --suffix _ji_v1 \
  --img-dir /home2/muskan.singh/benchmark/junctions

echo "Mode A done."

echo "Starting Mode C (no image)..."
python /home2/muskan.singh/code/eval.py     --model internvl3     --category junctions     --manifest /home2/muskan.singh/benchmark/junctions/ji_manifest_v1.json     --suffix _ji_v1 \
  --img-dir /home2/muskan.singh/benchmark/junctions     --no-image

echo "Mode C done."
