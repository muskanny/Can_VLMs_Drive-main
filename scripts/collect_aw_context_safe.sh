#!/bin/bash

set -euo pipefail

# ============================================================
# SAFE Adverse Weather Context Collector
# This script ONLY copies files into a new timestamped folder.
# It does NOT delete, move, rename, or modify existing files.
# ============================================================

USER_HOME="/home2/muskan.singh"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OUT_DIR="${USER_HOME}/aw_context_bundle_${TIMESTAMP}"
ARCHIVE="${USER_HOME}/aw_context_bundle_${TIMESTAMP}.tar.gz"

echo "============================================================"
echo "SAFE Adverse Weather Context Collector"
echo "============================================================"
echo "Output folder : ${OUT_DIR}"
echo "Archive file  : ${ARCHIVE}"
echo ""
echo "This script only copies files."
echo "It will NOT modify existing benchmark/results/code files."
echo "============================================================"

# ------------------------------------------------------------
# Create output structure
# ------------------------------------------------------------

mkdir -p "${OUT_DIR}"

mkdir -p "${OUT_DIR}/00_inventory"
mkdir -p "${OUT_DIR}/01_manifests"
mkdir -p "${OUT_DIR}/02_results_original_v2"
mkdir -p "${OUT_DIR}/03_results_mode_c"
mkdir -p "${OUT_DIR}/04_results_counterfactual"
mkdir -p "${OUT_DIR}/05_analysis_json"
mkdir -p "${OUT_DIR}/06_metadata"
mkdir -p "${OUT_DIR}/07_code"
mkdir -p "${OUT_DIR}/08_scripts"
mkdir -p "${OUT_DIR}/09_logs"
mkdir -p "${OUT_DIR}/10_samples"

# ------------------------------------------------------------
# Helper function: safe copy if file exists
# ------------------------------------------------------------

copy_if_exists () {
    SRC="$1"
    DEST="$2"

    if [ -f "$SRC" ]; then
        cp -n "$SRC" "$DEST"
        echo "Copied: $SRC"
    else
        echo "Missing, skipped: $SRC"
    fi
}

# ------------------------------------------------------------
# 0. Inventory of important directories
# ------------------------------------------------------------

echo ""
echo "[0/9] Creating directory inventories..."

{
    echo "Inventory created on: $(date)"
    echo ""
    echo "===== /home2/muskan.singh/benchmark/adverse_weather ====="
    find "${USER_HOME}/benchmark/adverse_weather" -maxdepth 5 -type f 2>/dev/null | sort || true
    echo ""
    echo "===== /home2/muskan.singh/results ====="
    find "${USER_HOME}/results" -maxdepth 1 -type f 2>/dev/null | sort || true
    echo ""
    echo "===== /home2/muskan.singh/code ====="
    find "${USER_HOME}/code" -maxdepth 1 -type f 2>/dev/null | sort || true
    echo ""
    echo "===== /home2/muskan.singh/scripts ====="
    find "${USER_HOME}/scripts" -maxdepth 1 -type f 2>/dev/null | sort || true
    echo ""
    echo "===== /home2/muskan.singh/logs ====="
    find "${USER_HOME}/logs" -maxdepth 1 -type f 2>/dev/null | sort || true
} > "${OUT_DIR}/00_inventory/full_inventory.txt"

# ------------------------------------------------------------
# 1. Manifests
# ------------------------------------------------------------

echo ""
echo "[1/9] Copying manifest files..."

copy_if_exists "${USER_HOME}/aw_manifest_v2.json" "${OUT_DIR}/01_manifests/"
copy_if_exists "${USER_HOME}/benchmark/adverse_weather/aw_manifest_v2.json" "${OUT_DIR}/01_manifests/"
copy_if_exists "${USER_HOME}/benchmark/adverse_weather/aw_manifest_cf_jarvis.json" "${OUT_DIR}/01_manifests/"

# also copy any extra AW manifest-like files
find "${USER_HOME}/benchmark/adverse_weather" \
    -maxdepth 2 \
    -type f \
    \( -name "aw_manifest*.json" -o -name "*manifest*.json" \) \
    -exec cp -n {} "${OUT_DIR}/01_manifests/" \; 2>/dev/null || true

# ------------------------------------------------------------
# 2. Original v2 inference results: Mode A
# ------------------------------------------------------------

echo ""
echo "[2/9] Copying original v2 Mode A result CSVs..."

find "${USER_HOME}/results" \
    -maxdepth 1 \
    -type f \
    -name "adverse_weather_*_v2.csv" \
    ! -name "*noimage*" \
    -exec cp -n {} "${OUT_DIR}/02_results_original_v2/" \; 2>/dev/null || true

# ------------------------------------------------------------
# 3. Mode C / no-image results
# ------------------------------------------------------------

echo ""
echo "[3/9] Copying Mode C / no-image result CSVs..."

find "${USER_HOME}/results" \
    -maxdepth 1 \
    -type f \
    \( -name "adverse_weather_*_noimage_v2.csv" -o -name "adverse_weather_*_noimage_cf.csv" \) \
    -exec cp -n {} "${OUT_DIR}/03_results_mode_c/" \; 2>/dev/null || true

# ------------------------------------------------------------
# 4. Counterfactual result CSVs
# ------------------------------------------------------------

echo ""
echo "[4/9] Copying counterfactual result CSVs..."

find "${USER_HOME}/results" \
    -maxdepth 1 \
    -type f \
    -name "adverse_weather_*_cf.csv" \
    ! -name "*noimage*" \
    -exec cp -n {} "${OUT_DIR}/04_results_counterfactual/" \; 2>/dev/null || true

# also copy any other cf-related AW CSVs
find "${USER_HOME}/results" \
    -maxdepth 1 \
    -type f \
    \( -name "*aw*cf*.csv" -o -name "*adverse*cf*.csv" \) \
    -exec cp -n {} "${OUT_DIR}/04_results_counterfactual/" \; 2>/dev/null || true

# ------------------------------------------------------------
# 5. Analysis JSON files
# ------------------------------------------------------------

echo ""
echo "[5/9] Copying existing analysis JSON files..."

copy_if_exists "${USER_HOME}/results/aw_linguistic_analysis.json" "${OUT_DIR}/05_analysis_json/"
copy_if_exists "${USER_HOME}/results/aw_cf_analysis.json" "${OUT_DIR}/05_analysis_json/"

find "${USER_HOME}/results" \
    -maxdepth 1 \
    -type f \
    \( -name "*aw*analysis*.json" -o -name "*linguistic*.json" -o -name "*scs*.json" \) \
    -exec cp -n {} "${OUT_DIR}/05_analysis_json/" \; 2>/dev/null || true

# -----------------------------------------------
# 6. Met
# -----------------------------------

echo 

find "${USER_HOME}/benchmark/adverse_weather" \
    -maxdepth 
    -type f \
    \( -name "*.csv" -o -
    -exec cp -n {} "${OUT_DIR}/06_metadata/" \; 2>/dev/null ||

# -----
# 7. Code files
# ------------------------------------------------------------

echo ""
echo "[7/9] Copying relevant Python code..."

copy_if_exists "${USER_HOME}/code/eval.py" "${OUT_DIR}/07_code/"
copy_if_exists "${USER_HOME}/code/analyze_aw_linguistic.py" "${OUT_DIR}/07_code/"
copy_if_exists "${USER_HOME}/code/analyze_aw_cf.py" "${OUT_DIR}/07_code/"

find "${USER_HOME}/code" \
    -maxdepth 1 \
    -type f \
    \( -name "*aw*.py" -o -name "*cf*.py" -o -name "*linguistic*.py" \) \
    -exec cp -n {} "${OUT_DIR}/07_code/" \; 2>/dev/null || true

# ------------------------------------------------------------
# 8. SLURM scripts
# ------------------------------------------------------------

echo ""
echo "[8/9] Copying relevant SLURM scripts..."

find "${USER_HOME}/scripts" \
    -maxdepth 1 \
    -type f \
    \( -name "run_*_aw_v2.sh" -o -name "run_*_aw_cf.sh" -o -name "*aw*.sh" -o -name "*cf*.sh" \) \
    -exec cp -n {} "${OUT_DIR}/08_scripts/" \; 2>/dev/null || true

# ------------------------------------------------------------
# 9. Logs and samples
# ------------------------------------------------------------

echo ""
echo "[9/9] Copying relevant logs and creating samples..."

find "${USER_HOME}/logs" \
    -maxdepth 1 \
    -type f \
    \( -name "*aw*.out" -o -name "*aw*.err" -o -name "*cf*.out" -o -name "*cf*.err" \) \
    -exec cp -n {} "${OUT_DIR}/09_logs/" \; 2>/dev/null || true

# Create small CSV previews
for CSV_FILE in \
    "${OUT_DIR}/02_results_original_v2"/*.csv \
    "${OUT_DIR}/03_results_mode_c"/*.csv \
    "${OUT_DIR}/04_results_counterfactual"/*.csv
do
    if [ -f "$CSV_FILE" ]; then
        BASENAME=$(basename "$CSV_FILE")
        head -n 25 "$CSV_FILE" > "${OUT_DIR}/10_samples/sample_${BASENAME}"
    fi
done

# Create small manifest previews
for JSON_FILE in "${OUT_DIR}/01_manifests"/*.json
do
    if [ -f "$JSON_FILE" ]; then
        BASENAME=$(basename "$JSON_FILE")
        python - "$JSON_FILE" "${OUT_DIR}/10_samples/sample_${BASENAME}" <<'PY'
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
        sample = dict(list(data.items())[:5])
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

# ------------------------------------------------------------
# README
# ------------------------------------------------------------

cat > "${OUT_DIR}/README.txt" <<README
Adverse Weather Context Bundle
==============================

Created on:
$(date)

This bundle was created using:
collect_aw_context_safe.sh

Safety note:
This script only copied files into a new folder.
It did not delete, move, rename, or modify any existing files.

Folders:

00_inventory/
- full_inventory.txt
- Directory listing of relevant Ada folders.

01_manifests/
- aw_manifest_v2.json
- aw_manifest_cf_jarvis.json
- other adverse-weather manifest files if found.

02_results_original_v2/
- Mode A image-based adverse weather v2 inference CSVs.

03_results_mode_c/
- Mode C / no-image inference CSVs.

04_results_counterfactual/
- Counterfactual adverse weather inference CSVs.

05_analysis_json/
- Existing analysis outputs such as aw_linguistic_analysis.json and aw_cf_analysis.json.

06_metadata/
- Metadata CSV/JSON files from benchmark/adverse_weather.

07_code/
- eval.py and adverse-weather analysis scripts.

08_scripts/
- SLURM scripts for AW v2 and AW CF runs.

09_logs/
- Relevant SLURM output/error logs.

10_samples/
- First 25 rows of CSV files and small JSON previews.

Important:
Full images are not included to keep the archive small.
Selected qualitative images can be collected later if needed.
README

# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Collection summary"
echo "============================================================"

du -sh "${OUT_DIR}" || true

echo ""
echo "Files copied by section:"
for DIR in "${OUT_DIR}"/*
do
    if [ -d "$DIR" ]; then
        COUNT=$(find "$DIR" -type f | wc -l)
        echo "$(basename "$DIR"): ${COUNT} files"
    fi
done

echo ""
echo "Creating archive..."
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
echo "============================================================"
