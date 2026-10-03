#!/bin/bash
set -euo pipefail

USER_HOME="/home2/muskan.singh"
RESULTS_DIR="${USER_HOME}/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="${USER_HOME}/phase2_bundle_${TIMESTAMP}"
ARCHIVE="${USER_HOME}/phase2_bundle_${TIMESTAMP}.tar.gz"

echo "============================================================"
echo "Phase 2 Bundle Collector"
echo "Output: ${OUT_DIR}"
echo "Archive: ${ARCHIVE}"
echo "============================================================"

mkdir -p "${OUT_DIR}/phase2_aw"
mkdir -p "${OUT_DIR}/phase2_ji"
mkdir -p "${OUT_DIR}/phase1_moondream_verbose"
mkdir -p "${OUT_DIR}/manifests"
mkdir -p "${OUT_DIR}/inventory"

copy_file() {
    if [ -f "$1" ]; then
        cp -n "$1" "$2"
        echo "Copied: $(basename $1)"
    else
        echo "MISSING: $1"
    fi
}

echo ""
echo "[1/5] Phase 2 AW results..."
copy_file "${RESULTS_DIR}/phase2_aw_llava_ov.csv"              "${OUT_DIR}/phase2_aw/"
copy_file "${RESULTS_DIR}/phase2_aw_llava_ov_noimage.csv"      "${OUT_DIR}/phase2_aw/"
copy_file "${RESULTS_DIR}/phase2_aw_internvl3.csv"             "${OUT_DIR}/phase2_aw/"
copy_file "${RESULTS_DIR}/phase2_aw_internvl3_noimage.csv"     "${OUT_DIR}/phase2_aw/"

echo ""
echo "[2/5] Phase 2 JI results..."
copy_file "${RESULTS_DIR}/phase2_ji_llava_ov.csv"              "${OUT_DIR}/phase2_ji/"
copy_file "${RESULTS_DIR}/phase2_ji_llava_ov_noimage.csv"      "${OUT_DIR}/phase2_ji/"
copy_file "${RESULTS_DIR}/phase2_ji_internvl3.csv"             "${OUT_DIR}/phase2_ji/"
copy_file "${RESULTS_DIR}/phase2_ji_internvl3_noimage.csv"     "${OUT_DIR}/phase2_ji/"

echo ""
echo "[3/5] Phase 1 Moondream verbose (AW + JI)..."
copy_file "${RESULTS_DIR}/adverse_weather_moondream_v2.csv"        "${OUT_DIR}/phase1_moondream_verbose/"
copy_file "${RESULTS_DIR}/adverse_weather_moondream_noimage_v2.csv" "${OUT_DIR}/phase1_moondream_verbose/"
copy_file "${RESULTS_DIR}/junctions_moondream_ji_v1_fixed.csv"     "${OUT_DIR}/phase1_moondream_verbose/"
copy_file "${RESULTS_DIR}/junctions_moondream_noimage_ji_v1.csv"   "${OUT_DIR}/phase1_moondream_verbose/"

echo ""
echo "[4/5] Manifests..."
copy_file "${USER_HOME}/benchmark/adverse_weather/aw_phase2_manifest_extended.json" "${OUT_DIR}/manifests/"
copy_file "${USER_HOME}/benchmark/junctions/ji_phase2_manifest.json"                "${OUT_DIR}/manifests/"

echo ""
echo "[5/5] Inventory..."
{
    echo "===== row counts ====="
    for f in \
        "${OUT_DIR}/phase2_aw"/*.csv \
        "${OUT_DIR}/phase2_ji"/*.csv \
        "${OUT_DIR}/phase1_moondream_verbose"/*.csv
    do
        [ -f "$f" ] && echo "$(wc -l < "$f")  $(basename "$f")"
    done
} > "${OUT_DIR}/inventory/row_counts.txt"

{
    echo "===== headers ====="
    for f in \
        "${OUT_DIR}/phase2_aw"/*.csv \
        "${OUT_DIR}/phase2_ji"/*.csv
    do
        if [ -f "$f" ]; then
            echo ""
            echo "----- $(basename "$f") -----"
            head -n 1 "$f"
        fi
    done
} > "${OUT_DIR}/inventory/headers.txt"

echo ""
echo "Compressing..."
tar -czf "${ARCHIVE}" -C "${USER_HOME}" "$(basename "${OUT_DIR}")"

echo ""
echo "============================================================"
echo "DONE"
echo "Archive: ${ARCHIVE}"
echo "Size: $(ls -lh "${ARCHIVE}" | awk '{print $5}')"
echo ""
echo "Row counts:"
cat "${OUT_DIR}/inventory/row_counts.txt"
echo "============================================================"
