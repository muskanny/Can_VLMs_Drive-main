#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition=ihub
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode095
#SBATCH --output=/home2/muskan.singh/logs/llava_ov_ns_%j.out
#SBATCH --job-name=llava_ov_ns
export HF_HOME=/home2/muskan.singh/hf_cache
export HF_HUB_DISABLE_XET=1
source /home2/muskan.singh/miniconda3/etc/profile.d/conda.sh
conda activate vlm

echo "Starting Mode A..."
python /home2/muskan.singh/code/eval.py \
    --model llava_ov \
    --category nuscenes \
    --manifest /home2/muskan.singh/benchmark/nuscenes/ns_manifest_200_extended.json \
    --suffix _ns \
    --img-dir /home2/muskan.singh/benchmark/nuscenes
echo "Mode A done."
echo "Starting Mode C..."
python /home2/muskan.singh/code/eval.py \
    --model llava_ov \
    --category nuscenes \
    --manifest /home2/muskan.singh/benchmark/nuscenes/ns_manifest_200_extended.json \
    --suffix _ns \
    --img-dir /home2/muskan.singh/benchmark/nuscenes \
    --no-image
echo "Mode C done."

