#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition=ihub
#SBATCH --gres=gpu:1
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3:00:00
#SBATCH --nodelist=gnode095
#SBATCH --output=/home2/muskan.singh/logs/phase2_aw_llava_ov_%j.out

export HF_HOME=/home2/muskan.singh/hf_cache
export TRANSFORMERS_CACHE=/home2/muskan.singh/hf_cache

source /home2/muskan.singh/miniconda3/bin/activate vlm
pip install transformers==4.57.6 -q

echo "=== Phase 2 AW Mode A + Linguistic + Counterfactual: llava_ov ==="
python /home2/muskan.singh/code/eval_phase2_aw.py --model llava_ov

echo "=== Phase 2 AW Mode C: llava_ov ==="
python /home2/muskan.singh/code/eval_phase2_aw.py --model llava_ov --no-image

echo "=== Deleting weights ==="
rm -rf /home2/muskan.singh/hf_cache/hub/models--*

echo "=== Done ==="
