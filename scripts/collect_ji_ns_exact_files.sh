#!/bin/bash

set -euo pipefail

USER_HOME="/home2/muskan.singh"
RESULTS_DIR="${USER_HOME}/results"
BENCH_JI="${USER_HOME}/benchmark/junctions"
BENCH_NS="${USER_HOME}/benchmark/nuscenes"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="${USER_HOME}/ji_ns_exact_bundle_${TIMESTAMP}"
ARCHIVE="${USER_HOME}/ji_ns_exact_bundle_${TIMESTAMP}.tar.gz"

echo "============================================================"
echo "Exact Junctions + nuScenes File Collector"
echo "============================================================"
echo "Output folder: ${OUT_DIR}"
echo "Archive: ${ARCHIVE}"
echo ""
echo "Safety: this script only copies exact files."
echo "It does not delete, move, rename, or edit any original files."
echo "============================================================"

mkdir -p "${OUT_DIR}/manifests"
mkdir -p "${OUT_DIR}/junctions_mode_a"
mkdir -p "${OUT_DIR}/junctions_mode_c"
mkdir -p "${OUT_DIR}/junctions_linguistic_mode_a"
mkdir -p "${OUT_DIR}/junctions_linguistic_mode_c"
mkdir -p "${OUT_DIR}/nuscenes_mode_a"
mkdir -p "${OUT_DIR}/nuscenes_mode_c"
mkdir -p "${OUT_DIR}/analysis_json"
mkdir -p "${OUT_DIR}/response_style_audit"
mkdir -p "${OUT_DIR}/code"
mkdir -p "${OUT_DIR}/scripts"
mkdir -p "${OUT_DIR}/inventory"
mkdir -p "${OUT_DIR}/samples"

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
echo "[1/9] Copying manifests..."

copy_file "${BENCH_JI}/ji_manifest_v1.json" "${OUT_DIR}/manifests/"
copy_file "${BENCH_JI}/ji_manifest_linguistic.json" "${OUT_DIR}/manifests/"
copy_file "${BENCH_NS}/ns_manifest_200_extended.json" "${OUT_DIR}/manifests/"

echo ""
echo "[2/9] Copying Junctions Mode A fixed CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/junctions_${model}_ji_v1_fixed.csv" "${OUT_DIR}/junctions_mode_a/"
done

echo ""
echo "[3/9] Copying Junctions Mode C CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/junctions_${model}_noimage_ji_v1.csv" "${OUT_DIR}/junctions_mode_c/"
done

echo ""
echo "[4/9] Copying Junctions linguistic Mode A CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/junctions_${model}_ji_ling.csv" "${OUT_DIR}/junctions_linguistic_mode_a/"
done

echo ""
echo "[5/9] Copying Junctions linguistic Mode C CSVs..."

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${RESULTS_DIR}/junctions_${model}_noimage_ji_ling.csv" "${OUT_DIR}/junctions_linguistic_mode_c/"
done

echo ""
echo "[6/9] Copying nuScenes Mode A CSVs..."

copy_file "${RESULTS_DIR}/nuscenes_moondream_ns.csv" "${OUT_DIR}/nuscenes_mode_a/"
copy_file "${RESULTS_DIR}/nuscenes_paligemma_ns_fixed.csv" "${OUT_DIR}/nuscenes_mode_a/"
copy_file "${RESULTS_DIR}/nuscenes_smolvlm_ns_fixed.csv" "${OUT_DIR}/nuscenes_mode_a/"
copy_file "${RESULTS_DIR}/nuscenes_llava_ov_ns.csv" "${OUT_DIR}/nuscenes_mode_a/"
copy_file "${RESULTS_DIR}/nuscenes_internvl3_ns.csv" "${OUT_DIR}/nuscenes_mode_a/"

echo ""
echo "[7/9] Copying nuScenes Mode C CSVs..."

copy_file "${RESULTS_DIR}/nuscenes_moondream_noimage_ns.csv" "${OUT_DIR}/nuscenes_mode_c/"
copy_file "${RESULTS_DIR}/nuscenes_paligemma_noimage_ns.csv" "${OUT_DIR}/nuscenes_mode_c/"
copy_file "${RESULTS_DIR}/nuscenes_smolvlm_noimage_ns.csv" "${OUT_DIR}/nuscenes_mode_c/"
copy_file "${RESULTS_DIR}/nuscenes_llava_ov_noimage_ns.csv" "${OUT_DIR}/nuscenes_mode_c/"
copy_file "${RESULTS_DIR}/nuscenes_internvl3_noimage_ns.csv" "${OUT_DIR}/nuscenes_mode_c/"

echo ""
echo "[8/9] Copying analysis JSONs, response audit, code, scripts..."

copy_file "${RESULTS_DIR}/junctions_analysis.json" "${OUT_DIR}/analysis_json/"
copy_file "${RESULTS_DIR}/junctions_linguistic_analysis.json" "${OUT_DIR}/analysis_json/"

copy_file "${RESULTS_DIR}/response_style_audit/ji_ns_response_style_summary.csv" "${OUT_DIR}/response_style_audit/"
copy_file "${RESULTS_DIR}/response_style_audit/ji_ns_response_style_samples.csv" "${OUT_DIR}/response_style_audit/"
copy_file "${RESULTS_DIR}/response_style_audit/ji_ns_response_style_summary.json" "${OUT_DIR}/response_style_audit/"

copy_file "${USER_HOME}/code/eval.py" "${OUT_DIR}/code/"
copy_file "${USER_HOME}/code/analyze_junctions.py" "${OUT_DIR}/code/"
copy_file "${USER_HOME}/code/analyze_ji_linguistic.py" "${OUT_DIR}/code/"
copy_file "${USER_HOME}/code/fix_extractor_junctions.py" "${OUT_DIR}/code/"
copy_file "${USER_HOME}/code/check_ji_ns_response_style.py" "${OUT_DIR}/code/"

for model in moondream paligemma smolvlm llava_ov internvl3
do
    copy_file "${USER_HOME}/scripts/run_${model}_ji_v1.sh" "${OUT_DIR}/scripts/"
    copy_file "${USER_HOME}/scripts/run_${model}_ji_v2.sh" "${OUT_DIR}/scripts/"
    copy_file "${USER_HOME}/scripts/run_${model}_ns.sh" "${OUT_DIR}/scripts/"
done

echo ""
echo "[9/9] Creating inventory and samples..."

cat > "${OUT_DIR}/inventory/README.txt" <<README
Junctions + nuScenes Exact Bundle
=================================

Created on:
$(date)

This bundle contains canonical files for Phase 1 analysis of:
1. Junctions & Intersections
2. Junctions linguistic variation
3. nuScenes spatial reasoning

Safety note:
The collector only copied exact files. It did not delete, move, rename, or edit original files.
README

{
    echo "===== row counts ====="
    for f in \
        "${OUT_DIR}/junctions_mode_a"/*.csv \
        "${OUT_DIR}/junctions_mode_c"/*.csv \
        "${OUT_DIR}/junctions_linguistic_mode_a"/*.csv \
        "${OUT_DIR}/junctions_linguistic_mode_c"/*.csv \
        "${OUT_DIR}/nuscenes_mode_a"/*.csv \
        "${OUT_DIR}/nuscenes_mode_c"/*.csv
    do
        if [ -f "$f" ]; then
            echo "$(wc -l < "$f")  $(basename "$f")"
        fi
    done
} > "${OUT_DIR}/inventory/row_counts.txt"

{
    echo "===== csv headers ====="
    for f in \
        "${OUT_DIR}/junctions_mode_a"/*.csv \
        "${OUT_DIR}/junctions_mode_c"/*.csv \
        "${OUT_DIR}/junctions_linguistic_mode_a"/*.csv \
        "${OUT_DIR}/junctions_linguistic_mode_c"/*.csv \
        "${OUT_DIR}/nuscenes_mode_a"/*.csv \
        "${OUT_DIR}/nuscenes_mode_c"/*.csv
    do
        if [ -f "$f" ]; then
            echo ""
            echo "----- $(basename "$f") -----"
            head -n 1 "$f"
        fi
    done
} > "${OUT_DIR}/inventory/headers.txt"

for f in \
    "${OUT_DIR}/junctions_mode_a"/*.csv \
    "${OUT_DIR}/junctions_mode_c"/*.csv \
    "${OUT_DIR}/junctions_linguistic_mode_a"/*.csv \
    "${OUT_DIR}/junctions_linguistic_mode_c"/*.csv \
    "${OUT_DIR}/nuscenes_mode_a"/*.csv \
    "${OUT_DIR}/nuscenes_mode_c"/*.csv
do
    if [ -f "$f" ]; then
        base=$(basename "$f")
        head -n 25 "$f" > "${OUT_DIR}/samples/sample_${base}"
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
