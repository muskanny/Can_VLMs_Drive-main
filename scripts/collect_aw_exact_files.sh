#!/bin/bash

set -euo pipefail

USER_HOME="/home2/muskan.singh"
RESULTS_DIR="${USER_HOME}/results"
BENCH_DIR="${USER_HOME}/benchmark/adverse_weather"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="${USER_HOME}/aw_exact_bundle_${TIMESTAMP}"
ARCHIVE="${USER_HOME}/aw_exact_bundle_${TIMESTAMP}.tar.gz"

echo "============================================================"
echo "Exact Adverse Weather File Collector"
echo "============================================================"
echo "Output folder: ${OUT_DIR}"
echo "Archive: ${ARCHIVE}"
echo ""
echo "Safety: this script only copies exact files."
echo "It does not delete, move, rename, or edit any original files."
echo "============================================================"

mkdir -p "${OUT_DIR}/manifests"
mkdir -p "${OUT_DIR}/results_v2_mode_a"
mkdir -p "${OUT_DIR}/results_v2_mode_c"
mkdir -p "${OUT_DIR}/results_cf_mode_a"
mkdir -p "${OUT_DIR}/results_cf_mode_c"
mkdir -p "${OUT_DIR}/analysis_json"
mkdir -p "${OUT_DIR}/code"
mkdir -p "${OUT_DIR}/scripts"
mkdir -p "${OUT_DIR}/samples"
mkdir -p "${OUT_DIR}/inventory"

copy_file () {
    SRC="$1"
    DEST="$2"

    if [ -f "$SRC" ]; then
        cp -n "$SRC" "$DEST"
        echo "Copied: $SRC"
    else
        echo "MISSING: $SRC"
    fi
}

echo ""
echo "[1/8] Copying manifests..."

copy_file "${USER_HOME}/aw_manifest_v2.json" "${OUT_DIR}/manifests/"
copy_file "${BENCH_DIR}/aw_manifest_cf_jarvis.json" "${OUT_DIR}/manifests/"

echo ""
echo "[2/8] Copying final v2 Mode A result CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/adverse_weather_${model}_v2.csv" "${OUT_DIR}/results_v2_mode_a/"
done

echo ""
echo "[3/8] Copying final v2 Mode C / no-image CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/adverse_weather_${model}_noimage_v2.csv" "${OUT_DIR}/results_v2_mode_c/"
done

echo ""
echo "[4/8] Copying counterfactual Mode A result CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/adverse_weather_${model}_cf.csv" "${OUT_DIR}/results_cf_mode_a/"
done

echo ""
echo "[5/8] Copying counterfactual Mode C / no-image CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/adverse_weather_${model}_noimage_cf.csv" "${OUT_DIR}/results_cf_mode_c/"
done

echo ""
echo "[6/8] Copying existing analysis JSON files..."

copy_file "${RESULTS_DIR}/aw_linguistic_analysis.json" "${OUT_DIR}/analysis_json/"
copy_file "${RESULTS_DIR}/aw_cf_analysis.json" "${OUT_DIR}/analysis_json/"

echo ""
echo "[7/8] Copying relevant code and SLURM scripts..."

copy_file "${USER_HOME}/code/eval.py" "${OUT_DIR}/code/"
copy_file "${USER_HOME}/code/analyze_aw_linguistic.py" "${OUT_DIR}/code/"
copy_file "${USER_HOME}/code/analyze_aw_cf.py" "${OUT_DIR}/code/"

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${USER_HOME}/scripts/run_${model}_aw_v2.sh" "${OUT_DIR}/scripts/"
    copy_file "${USER_HOME}/scripts/run_${model}_aw_cf.sh" "${OUT_DIR}/scripts/"
done

echo ""
echo "[8/8] Creating inventory and samples..."

{
    echo "Adverse Weather exact bundle created on: $(date)"
    echo ""
    echo "===== final files expected ====="
    echo "Manifests:"
    echo "- aw_manifest_v2.json"
    echo "- aw_manifest_cf_jarvis.json"
    echo ""
    echo "Models:"
    echo "- moondream"
    echo "- paligemma"
    echo "- smolvlm"
    echo "- llava_ov"
    echo "- internvl3"
    echo ""
    echo "For each model:"
    echo "- adverse_weather_<model>_v2.csv"
    echo "- adverse_weather_<model>_noimage_v2.csv"
    echo "- adverse_weather_<model>_cf.csv"
    echo "- adverse_weather_<model>_noimage_cf.csv"
    echo ""
    echo "Analysis JSON:"
    echo "- aw_linguistic_analysis.json"
    echo "- aw_cf_analysis.json"
} > "${OUT_DIR}/inventory/README.txt"

echo ""
echo "Creating row-count inventory..."

{
    echo "===== row counts ====="
    for f in \
        "${OUT_DIR}/results_v2_mode_a"/*.csv \
        "${OUT_DIR}/results_v2_mode_c"/*.csv \
        "${OUT_DIR}/results_cf_mode_a"/*.csv \
        "${OUT_DIR}/results_cf_mode_c"/*.csv
    do
        if [ -f "$f" ]; then
            echo "$(wc -l < "$f")  $(basename "$f")"
        fi
    done
} > "${OUT_DIR}/inventory/row_counts.txt"

echo ""
echo "Creating CSV samples..."

for f in \
    "${OUT_DIR}/results_v2_mode_a"/*.csv \
    "${OUT_DIR}/results_v2_mode_c"/*.csv \
    "${OUT_DIR}/results_cf_mode_a"/*.csv \
    "${OUT_DIR}/results_cf_mode_c"/*.csv
do
    if [ -f "$f" ]; then
        base=$(basename "$f")
        head -n 25 "$f" > "${OUT_DIR}/samples/sample_${base}"
    fi
done

echo ""
echo "Creating JSON samples..."

for f in "${OUT_DIR}/manifests"/*.json "${OUT_DIR}/analysis_json"/*.json
do
    if [ -f "$f" ]; then
        base=$(basename "$f")
        python - "$f" "${OUT_DIR}/samples/sample_${base}" <<'PY'
import json
import sys

src = sys.argv[1]
dst = sys.argv[2]

try:
    with open(src, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        sample = data[:2]
    elif isinstance(data, dict):
        sample = dict(list(data.items())[:3])
    else:
        sample = data

    with open(dst, "w") as f:
        json.dump(sample, f, indent=2)

except Exception as e:
    with open(dst, "w") as f:
        f.write(f"Could not parse JSON: {e}\n")
PY
    fi
done

echo ""
echo "Compressing bundle..."

tar -czf "${ARCHIVE}" -C "${USER_HOME}" "$(basename "${OUT_DIR}")"

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Output folder:"
echo "${OUT_DIR}"
echo ""
echo "Archive:"
echo "${ARCHIVE}"
echo ""
echo "Archive size:"
ls -lh "${ARCHIVE}"
echo ""
echo "Row counts:"
cat "${OUT_DIR}/inventory/row_counts.txt"
echo "============================================================"
