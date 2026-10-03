#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition=ihub
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode093
#SBATCH --output=/home2/muskan.singh/logs/smolvlm_ji_v2_%j.out
#SBATCH --job-name=smolvlm_ji_v2
export HF_HOME=/home2/muskan.singh/hf_cache
export HF_HUB_DISABLE_XET=1
source /home2/muskan.singh/miniconda3/etc/profile.d/conda.sh
conda activate vlm

echo "Starting Mode A (with image)..."
python /home2/muskan.singh/code/eval.py \
    --model smolvlm \
    --category junctions \
    --manifest /home2/muskan.singh/benchmark/junctions/ji_manifest_linguistic.json \
    --suffix _ji_ling \
    --img-dir /home2/muskan.singh/benchmark/junctions
echo "Mode A done."
echo "Starting Mode C (no image)..."
python /home2/muskan.singh/code/eval.py \
    --model smolvlm \
    --category junctions \
    --manifest /home2/muskan.singh/benchmark/junctions/ji_manifest_linguistic.json \
    --suffix _ji_ling \
    --img-dir /home2/muskan.singh/benchmark/junctions \
    --no-image
echo "Mode C done."

