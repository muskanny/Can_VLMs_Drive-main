#!/usr/bin/env python3

import csv
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

RESULTS_DIR = Path("/home2/muskan.singh/results")
OUT_DIR = RESULTS_DIR / "response_style_audit"
OUT_DIR.mkdir(exist_ok=True)

JUNCTION_MODELS = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]
NS_MODELS = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]

files = []

# Junctions original + no-image + linguistic + linguistic no-image
for model in JUNCTION_MODELS:
    files.append({
        "category": "junctions",
        "mode": "Mode A",
        "analysis_group": "ji_v1",
        "model": model,
        "path": RESULTS_DIR / f"junctions_{model}_ji_v1_fixed.csv",
    })
    files.append({
        "category": "junctions",
        "mode": "Mode C",
        "analysis_group": "ji_v1_noimage",
        "model": model,
        "path": RESULTS_DIR / f"junctions_{model}_noimage_ji_v1.csv",
    })
    files.append({
        "category": "junctions",
        "mode": "Mode A",
        "analysis_group": "ji_linguistic",
        "model": model,
        "path": RESULTS_DIR / f"junctions_{model}_ji_ling.csv",
    })
    files.append({
        "category": "junctions",
        "mode": "Mode C",
        "analysis_group": "ji_linguistic_noimage",
        "model": model,
        "path": RESULTS_DIR / f"junctions_{model}_noimage_ji_ling.csv",
    })

# nuScenes spatial files: tries both normal and fixed names
for model in NS_MODELS:
    for name in [
        f"nuscenes_{model}_ns.csv",
        f"nuscenes_{model}_ns_fixed.csv",
    ]:
        p = RESULTS_DIR / name
        if p.exists():
            files.append({
                "category": "nuscenes_spatial",
                "mode": "Mode A",
                "analysis_group": "ns_spatial",
                "model": model,
                "path": p,
            })

YES_RE = re.compile(r"^\s*(?:\*\*)?\s*yes\s*(?:\*\*)?\s*[\.\!\,]?\s*$", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(?:\*\*)?\s*no\s*(?:\*\*)?\s*[\.\!\,]?\s*$", re.IGNORECASE)

START_YES_RE = re.compile(r"^\s*(?:\*\*)?\s*yes\b", re.IGNORECASE)
START_NO_RE = re.compile(r"^\s*(?:\*\*)?\s*no\b", re.IGNORECASE)

REFUSAL_PATTERNS = [
    "unanswerable",
    "cannot answer",
    "can't answer",
    "can not answer",
    "not enough information",
    "insufficient information",
    "no image",
    "without the image",
    "without an image",
    "as a language model",
    "i cannot determine",
    "i can't determine",
    "unable to determine",
    "not possible to determine",
    "cannot be determined",
    "can't be determined",
    "sorry",
]

BLANK_IMAGE_PATTERNS = [
    "black image",
    "blank image",
    "black background",
    "blank background",
    "empty image",
    "no visible scene",
    "nothing visible",
    "solid black",
]


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def classify_response(resp):
    text = clean_text(resp)
    low = text.lower()

    if text == "":
        return "empty"

    if YES_RE.match(text) or NO_RE.match(text):
        return "strict_yes_no"

    if any(p in low for p in REFUSAL_PATTERNS):
        return "refusal_or_unanswerable"

    if any(p in low for p in BLANK_IMAGE_PATTERNS):
        return "blank_image_description"

    if START_YES_RE.match(text) or START_NO_RE.match(text):
        return "yes_no_with_explanation"

    return "verbose_or_other"


def pct(n, d):
    if d == 0:
        return 0.0
    return round(100.0 * n / d, 2)


summary_rows = []
sample_rows = []
missing_files = []

for item in files:
    path = item["path"]

    if not path.exists():
        missing_files.append(str(path))
        continue

    style_counts = Counter()
    ans_counts = Counter()
    total = 0

    # samples_by_style limits saved samples per style
    samples_by_style = defaultdict(int)

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if "full_response" not in reader.fieldnames:
                print(f"[WARN] full_response column missing in {path.name}")
                continue

            for row in reader:
                total += 1

                response = clean_text(row.get("full_response", ""))
                extracted = clean_text(row.get("extracted_ans", "")).lower()

                style = classify_response(response)

                style_counts[style] += 1
                ans_counts[extracted] += 1

                if style != "strict_yes_no" and samples_by_style[style] < 5:
                    sample_rows.append({
                        "category": item["category"],
                        "analysis_group": item["analysis_group"],
                        "mode": item["mode"],
                        "model": item["model"],
                        "file": path.name,
                        "response_style": style,
                        "q_id": row.get("q_id", ""),
                        "question": row.get("question", ""),
                        "ground_truth": row.get("ground_truth", row.get("gt", "")),
                        "extracted_ans": row.get("extracted_ans", ""),
                        "correct": row.get("correct", ""),
                        "full_response": response.replace("\n", " ")[:500],
                    })
                    samples_by_style[style] += 1

    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        continue

    summary_rows.append({
        "category": item["category"],
        "analysis_group": item["analysis_group"],
        "mode": item["mode"],
        "model": item["model"],
        "file": path.name,
        "rows": total,

        "strict_yes_no_n": style_counts["strict_yes_no"],
        "strict_yes_no_pct": pct(style_counts["strict_yes_no"], total),

        "yes_no_with_explanation_n": style_counts["yes_no_with_explanation"],
        "yes_no_with_explanation_pct": pct(style_counts["yes_no_with_explanation"], total),

        "refusal_or_unanswerable_n": style_counts["refusal_or_unanswerable"],
        "refusal_or_unanswerable_pct": pct(style_counts["refusal_or_unanswerable"], total),

        "blank_image_description_n": style_counts["blank_image_description"],
        "blank_image_description_pct": pct(style_counts["blank_image_description"], total),

        "verbose_or_other_n": style_counts["verbose_or_other"],
        "verbose_or_other_pct": pct(style_counts["verbose_or_other"], total),

        "empty_n": style_counts["empty"],
        "empty_pct": pct(style_counts["empty"], total),

        "extracted_yes_pct": pct(ans_counts["yes"], total),
        "extracted_no_pct": pct(ans_counts["no"], total),
        "extracted_unclear_pct": pct(ans_counts["unclear"], total),
    })

summary_csv = OUT_DIR / "ji_ns_response_style_summary.csv"
samples_csv = OUT_DIR / "ji_ns_response_style_samples.csv"
summary_json = OUT_DIR / "ji_ns_response_style_summary.json"

summary_fields = [
    "category",
    "analysis_group",
    "mode",
    "model",
    "file",
    "rows",
    "strict_yes_no_n",
    "strict_yes_no_pct",
    "yes_no_with_explanation_n",
    "yes_no_with_explanation_pct",
    "refusal_or_unanswerable_n",
    "refusal_or_unanswerable_pct",
    "blank_image_description_n",
    "blank_image_description_pct",
    "verbose_or_other_n",
    "verbose_or_other_pct",
    "empty_n",
    "empty_pct",
    "extracted_yes_pct",
    "extracted_no_pct",
    "extracted_unclear_pct",
]

sample_fields = [
    "category",
    "analysis_group",
    "mode",
    "model",
    "file",
    "response_style",
    "q_id",
    "question",
    "ground_truth",
    "extracted_ans",
    "correct",
    "full_response",
]

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

with open(samples_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=sample_fields)
    writer.writeheader()
    writer.writerows(sample_rows)

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary_rows, f, indent=2)

print("\n============================================================")
print("Junctions + nuScenes Response Style Audit Complete")
print("============================================================")
print(f"Summary CSV: {summary_csv}")
print(f"Samples CSV: {samples_csv}")
print(f"Summary JSON: {summary_json}")

print("\nMissing expected files:")
if missing_files:
    for m in missing_files:
        print("  -", m)
else:
    print("  None")

print("\nCompact summary:")
compact_cols = [
    "category",
    "analysis_group",
    "mode",
    "model",
    "rows",
    "strict_yes_no_pct",
    "yes_no_with_explanation_pct",
    "refusal_or_unanswerable_pct",
    "blank_image_description_pct",
    "verbose_or_other_pct",
    "extracted_yes_pct",
    "extracted_no_pct",
    "extracted_unclear_pct",
]

print(",".join(compact_cols))
for row in summary_rows:
    print(",".join(str(row.get(c, "")) for c in compact_cols))

print("\nSample non-strict responses:")
for row in sample_rows[:30]:
    print("-" * 80)
    print(f"{row['category']} | {row['analysis_group']} | {row['mode']} | {row['model']} | {row['response_style']}")
    print(f"Q-ID: {row['q_id']}")
    print(f"Extracted: {row['extracted_ans']} | Correct: {row['correct']}")
    print(f"Response: {row['full_response']}")
