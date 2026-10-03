"""
analyze_aw_linguistic.py
------------------------
Linguistic variation analysis for Adverse Weather category.
Requires: aw_manifest_v2.json (for variation_type + source_q_id mapping)
          adverse_weather_<model>_v2.csv (all 5 models)

Two analyses:
  1. PARAPHRASE CONSISTENCY — for each (image, original_q), did model give same
     answer on paraphrase variants? High consistency = model is stable.
     Low consistency = model is sensitive to surface wording.

  2. NEGATION ACCURACY — for each (image, original_q), did model correctly flip
     its answer on the negated variant? (GT is flipped for negated variants)
     Low negation accuracy = model fails to process negation.
     Connects directly to the Negation Bias category.

Output:
  - aw_linguistic_analysis.json
  - printed summary tables
"""

import csv
import json
import os
from collections import defaultdict

RESULTS_DIR  = "/home2/muskan.singh/results"
MANIFEST_PATH = "/home2/muskan.singh/aw_manifest_v2.json"
OUTPUT_DIR   = "/home2/muskan.singh/results"

MODELS = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]

# ── helpers ──────────────────────────────────────────────────────────────────

def norm_correct(val):
    return str(val).strip().lower() == "true"

def norm_ans(val):
    return str(val).strip().lower()

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def build_manifest_lookup(manifest_path):
    """
    Returns two dicts from manifest:
      q_meta[q_id] = {variation_type, source_q_id, question, gt}
      source_to_variants[source_q_id] = {
          'paraphrase': [q_id, ...],
          'negated':    [q_id, ...]
      }
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    q_meta = {}
    source_to_variants = defaultdict(lambda: {'paraphrase': [], 'negated': []})

    for entry in manifest:
        for q in entry['questions']:
            q_id = q['q_id']
            vtype = q.get('variation_type', 'original')
            src   = q.get('source_q_id', q_id)
            if q_id not in q_meta:
                q_meta[q_id] = {
                    'variation_type': vtype,
                    'source_q_id'   : src,
                    'question'      : q['question'],
                    'gt'            : q.get('gt', ''),
                    'use_for'       : q.get('use_for', ''),
                    'q_type'        : q.get('q_type', ''),
                }
            if vtype in ('paraphrase', 'negated') and q_id not in source_to_variants[src][vtype]:
                source_to_variants[src][vtype].append(q_id)

    return q_meta, source_to_variants

# ── per-model analysis ────────────────────────────────────────────────────────

def analyze_model(model, rows, q_meta, source_to_variants):
    """
    Returns:
      paraphrase_consistency: per source_q_id → % of (image, variant) pairs
                              where model gave same answer as on original
      negation_accuracy:      per source_q_id → % of (image, negated_variant)
                              pairs where model answer was correct
                              (correct = extracted_ans matches negated GT)
    """
    # Index rows by (image_id, q_id)
    ans_index = {}
    for r in rows:
        if norm_ans(r['extracted_ans']) in ('yes', 'no'):
            ans_index[(r['image_id'], r['q_id'])] = norm_ans(r['extracted_ans'])

    # Get all source (original) q_ids
    source_q_ids = [q_id for q_id, meta in q_meta.items()
                    if meta['variation_type'] == 'original'
                    and meta['use_for'] == 'main_eval']

    paraphrase_results = {}  # source_q_id → {consistent, total}
    negation_results   = {}  # source_q_id → {correct, total}

    for src_q_id in source_q_ids:
        para_variants = source_to_variants[src_q_id]['paraphrase']
        neg_variants  = source_to_variants[src_q_id]['negated']

        # All images that have an answer for the original question
        images_with_original = [img for (img, qid) in ans_index if qid == src_q_id]

        # --- Paraphrase consistency ---
        p_consistent = 0
        p_total = 0
        for img in images_with_original:
            orig_ans = ans_index.get((img, src_q_id))
            if orig_ans is None:
                continue
            for var_q_id in para_variants:
                var_ans = ans_index.get((img, var_q_id))
                if var_ans is not None:
                    p_total += 1
                    if var_ans == orig_ans:
                        p_consistent += 1

        paraphrase_results[src_q_id] = {
            'consistent': p_consistent,
            'total'     : p_total,
            'pct'       : round(p_consistent / p_total * 100, 1) if p_total else None
        }

        # --- Negation accuracy ---
        n_correct = 0
        n_total   = 0
        for img in images_with_original:
            for neg_q_id in neg_variants:
                neg_ans = ans_index.get((img, neg_q_id))
                if neg_ans is None:
                    continue
                # GT for negated variant is flipped from original
                orig_gt = q_meta[src_q_id]['gt']
                neg_gt  = 'no' if orig_gt == 'yes' else 'yes'
                n_total += 1
                if neg_ans == neg_gt:
                    n_correct += 1

        negation_results[src_q_id] = {
            'correct': n_correct,
            'total'  : n_total,
            'pct'    : round(n_correct / n_total * 100, 1) if n_total else None
        }

    return paraphrase_results, negation_results

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading manifest...")
    q_meta, source_to_variants = build_manifest_lookup(MANIFEST_PATH)

    source_q_ids = sorted([q_id for q_id, meta in q_meta.items()
                           if meta['variation_type'] == 'original'
                           and meta['use_for'] == 'main_eval'])

    all_results = {}

    for model in MODELS:
        path = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_v2.csv")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        rows = load_csv(path)
        para, neg = analyze_model(model, rows, q_meta, source_to_variants)
        all_results[model] = {
            'paraphrase_consistency': para,
            'negation_accuracy'     : neg,
        }
        print(f"  {model} done")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "aw_linguistic_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Print: Paraphrase Consistency ─────────────────────────────────────────
    print("\n" + "="*72)
    print("PARAPHRASE CONSISTENCY  (% of paraphrase variants model answered same as original)")
    print("Higher = more stable across surface-level rewording")
    print("="*72)

    header = f"{'Q_ID':<14} {'Type':<12}"
    for m in MODELS:
        if m in all_results:
            header += f" {m[:7]:>8}"
    header += f"  {'Avg':>6}"
    print(header)
    print("-" * len(header))

    for q_id in source_q_ids:
        q_type = q_meta[q_id]['q_type']
        row_str = f"{q_id:<14} {q_type:<12}"
        vals = []
        for m in MODELS:
            if m not in all_results:
                continue
            pct = all_results[m]['paraphrase_consistency'].get(q_id, {}).get('pct')
            row_str += f" {str(pct) if pct is not None else '—':>8}"
            if pct is not None:
                vals.append(pct)
        avg = round(sum(vals)/len(vals), 1) if vals else None
        row_str += f"  {str(avg):>6}"
        print(row_str)

    # Model-level paraphrase consistency
    print("\n--- Model-level paraphrase consistency (avg across all questions) ---")
    for m in MODELS:
        if m not in all_results:
            continue
        vals = [v['pct'] for v in all_results[m]['paraphrase_consistency'].values()
                if v['pct'] is not None]
        avg = round(sum(vals)/len(vals), 1) if vals else None
        print(f"  {m:<14} {avg}%")

    # ── Print: Negation Accuracy ──────────────────────────────────────────────
    print("\n" + "="*72)
    print("NEGATION ACCURACY  (% correct on negated variants — GT is flipped)")
    print("Lower = model fails to process negation (connects to Negation Bias category)")
    print("50% = chance level (random yes/no)")
    print("="*72)

    header2 = f"{'Q_ID':<14} {'Type':<12}"
    for m in MODELS:
        if m in all_results:
            header2 += f" {m[:7]:>8}"
    header2 += f"  {'Avg':>6}"
    print(header2)
    print("-" * len(header2))

    for q_id in source_q_ids:
        q_type = q_meta[q_id]['q_type']
        row_str = f"{q_id:<14} {q_type:<12}"
        vals = []
        for m in MODELS:
            if m not in all_results:
                continue
            pct = all_results[m]['negation_accuracy'].get(q_id, {}).get('pct')
            row_str += f" {str(pct) if pct is not None else '—':>8}"
            if pct is not None:
                vals.append(pct)
        avg = round(sum(vals)/len(vals), 1) if vals else None
        row_str += f"  {str(avg):>6}"
        print(row_str)

    # Model-level negation accuracy
    print("\n--- Model-level negation accuracy (avg across all questions) ---")
    for m in MODELS:
        if m not in all_results:
            continue
        vals = [v['pct'] for v in all_results[m]['negation_accuracy'].values()
                if v['pct'] is not None]
        avg = round(sum(vals)/len(vals), 1) if vals else None
        print(f"  {m:<14} {avg}%  {'← near chance' if avg and avg < 55 else ''}")

    print(f"\nDone. Full results: {out_path}")

if __name__ == "__main__":
    main()
