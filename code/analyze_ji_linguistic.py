"""
analyze_ji_linguistic.py
------------------------
Linguistic variation analysis for Junctions & Intersections category.
Requires: ji_manifest_linguistic.json (variation_type + source_q_id mapping)
          junctions_<model>_ji_v1_fixed.csv  (original questions, extractor-fixed)
          junctions_<model>_ji_ling.csv       (linguistic variants)

Two analyses:
  1. PARAPHRASE CONSISTENCY — for each (image, original_q), did model give same
     answer on paraphrase variants? High = stable. Low = sensitive to wording.

  2. NEGATION ACCURACY — for each (image, original_q), did model correctly flip
     its answer on the negated variant?
     Low negation accuracy = model fails to process negation.
     Connects directly to the Negation Bias category and AW linguistic findings.

Output:
  - junctions_linguistic_analysis.json
  - printed summary tables
"""

import csv
import json
import os
from collections import defaultdict

RESULTS_DIR   = "/home2/muskan.singh/results"
MANIFEST_PATH = "/home2/muskan.singh/benchmark/junctions/ji_manifest_linguistic.json"
OUTPUT_DIR    = "/home2/muskan.singh/results"

MODELS = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]

# ── helpers ───────────────────────────────────────────────────────────────────

def norm_ans(val):
    return str(val).strip().lower()

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def build_manifest_lookup(manifest_path):
    """
    Returns:
      q_meta[q_id] = {variation_type, source_q_id, question, gt, q_type, use_for}
      source_to_variants[source_q_id] = {
          'paraphrase': [q_id, ...],
          'negated':    [q_id, ...]
      }
    NOTE: ji_manifest_linguistic.json contains ONLY paraphrase and negated variants
    (originals are excluded). Original answers come from ji_v1_fixed CSVs.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    q_meta = {}
    source_to_variants = defaultdict(lambda: {'paraphrase': [], 'negated': []})

    for entry in manifest:
        for q in entry['questions']:
            q_id  = q['q_id']
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

def build_original_lookup(manifest_path):
    """
    Get all original q_ids and their GT from the v1 manifest
    (not the linguistic manifest which only has variants).
    """
    v1_path = "/home2/muskan.singh/benchmark/junctions/ji_manifest_v1.json"
    with open(v1_path) as f:
        manifest = json.load(f)

    orig_q_meta = {}
    for entry in manifest:
        for q in entry['questions']:
            q_id = q['q_id']
            if q_id not in orig_q_meta:
                orig_q_meta[q_id] = {
                    'variation_type': 'original',
                    'source_q_id'   : q_id,
                    'question'      : q['question'],
                    'gt'            : q.get('gt', ''),
                    'use_for'       : q.get('use_for', ''),
                    'q_type'        : q.get('q_type', ''),
                }
    return orig_q_meta

# ── per-model analysis ────────────────────────────────────────────────────────

def analyze_model(model, orig_rows, ling_rows, orig_q_meta, q_meta, source_to_variants):
    """
    orig_rows: from junctions_<model>_ji_v1_fixed.csv
    ling_rows: from junctions_<model>_ji_ling.csv

    Returns:
      paraphrase_results[src_q_id] = {consistent, total, pct}
      negation_results[src_q_id]   = {correct, total, pct}
    """
    # Index original answers by (image_id, q_id)
    ans_orig = {}
    for r in orig_rows:
        a = norm_ans(r.get('extracted_ans', ''))
        if a in ('yes', 'no'):
            ans_orig[(r['image_id'], r['q_id'])] = a

    # Index linguistic variant answers by (image_id, q_id)
    ans_ling = {}
    for r in ling_rows:
        a = norm_ans(r.get('extracted_ans', ''))
        if a in ('yes', 'no'):
            ans_ling[(r['image_id'], r['q_id'])] = a

    # Source q_ids = all original main_eval questions
    source_q_ids = [q_id for q_id, meta in orig_q_meta.items()
                    if meta['use_for'] == 'main_eval']

    paraphrase_results = {}
    negation_results   = {}

    for src_q_id in source_q_ids:
        para_variants = source_to_variants[src_q_id]['paraphrase']
        neg_variants  = source_to_variants[src_q_id]['negated']

        # All images that have an answer for the original question
        images_with_original = [img for (img, qid) in ans_orig if qid == src_q_id]

        # --- Paraphrase consistency ---
        p_consistent = 0
        p_total      = 0
        for img in images_with_original:
            orig_ans = ans_orig.get((img, src_q_id))
            if orig_ans is None:
                continue
            for var_q_id in para_variants:
                var_ans = ans_ling.get((img, var_q_id))
                if var_ans is not None:
                    p_total += 1
                    if var_ans == orig_ans:
                        p_consistent += 1

        paraphrase_results[src_q_id] = {
            'consistent': p_consistent,
            'total'     : p_total,
            'pct'       : round(p_consistent / p_total * 100, 1) if p_total else None,
        }

        # --- Negation accuracy ---
        n_correct = 0
        n_total   = 0
        for img in images_with_original:
            for neg_q_id in neg_variants:
                neg_ans = ans_ling.get((img, neg_q_id))
                if neg_ans is None:
                    continue
                orig_gt = orig_q_meta[src_q_id]['gt']
                neg_gt  = 'no' if orig_gt == 'yes' else 'yes'
                n_total += 1
                if neg_ans == neg_gt:
                    n_correct += 1

        negation_results[src_q_id] = {
            'correct': n_correct,
            'total'  : n_total,
            'pct'    : round(n_correct / n_total * 100, 1) if n_total else None,
        }

    return paraphrase_results, negation_results

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading manifests...")
    q_meta, source_to_variants = build_manifest_lookup(MANIFEST_PATH)
    orig_q_meta = build_original_lookup(MANIFEST_PATH)

    source_q_ids = sorted(orig_q_meta.keys())

    all_results = {}

    for model in MODELS:
        orig_path = os.path.join(RESULTS_DIR, f"junctions_{model}_ji_v1_fixed.csv")
        ling_path = os.path.join(RESULTS_DIR, f"junctions_{model}_ji_ling.csv")

        if not os.path.exists(orig_path):
            print(f"[SKIP] {orig_path} not found")
            continue
        if not os.path.exists(ling_path):
            print(f"[SKIP] {ling_path} not found")
            continue

        orig_rows = load_csv(orig_path)
        ling_rows = load_csv(ling_path)
        print(f"  {model}: {len(orig_rows)} original rows, {len(ling_rows)} ling rows")

        para, neg = analyze_model(
            model, orig_rows, ling_rows,
            orig_q_meta, q_meta, source_to_variants
        )
        all_results[model] = {
            'paraphrase_consistency': para,
            'negation_accuracy'     : neg,
        }

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "junctions_linguistic_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Print: Paraphrase Consistency ──────────────────────────────────────────
    print("\n" + "="*76)
    print("PARAPHRASE CONSISTENCY  (% paraphrase variants answered same as original)")
    print("Higher = stable across surface rewording")
    print("="*76)

    header = f"{'Q_ID':<14} {'Type':<14}"
    for m in MODELS:
        if m in all_results:
            header += f" {m[:7]:>8}"
    header += f"  {'Avg':>6}"
    print(header)
    print("-" * len(header))

    for q_id in source_q_ids:
        q_type = orig_q_meta[q_id]['q_type']
        row_str = f"{q_id:<14} {q_type:<14}"
        vals = []
        for m in MODELS:
            if m not in all_results:
                continue
            pct = all_results[m]['paraphrase_consistency'].get(q_id, {}).get('pct')
            row_str += f" {str(pct) if pct is not None else '—':>8}"
            if pct is not None:
                vals.append(pct)
        avg = round(sum(vals)/len(vals), 1) if vals else None
        row_str += f"  {str(avg) if avg is not None else '—':>6}"
        print(row_str)

    print("\n--- Model-level paraphrase consistency ---")
    for m in MODELS:
        if m not in all_results:
            continue
        vals = [v['pct'] for v in all_results[m]['paraphrase_consistency'].values()
                if v['pct'] is not None]
        avg = round(sum(vals)/len(vals), 1) if vals else None
        print(f"  {m:<14} {avg}%")

    # ── Print: Negation Accuracy ───────────────────────────────────────────────
    print("\n" + "="*76)
    print("NEGATION ACCURACY  (% correct on negated variants — GT is flipped)")
    print("50% = chance level.  Lower = model fails negation.")
    print("="*76)

    header2 = f"{'Q_ID':<14} {'Type':<14}"
    for m in MODELS:
        if m in all_results:
            header2 += f" {m[:7]:>8}"
    header2 += f"  {'Avg':>6}"
    print(header2)
    print("-" * len(header2))

    for q_id in source_q_ids:
        q_type = orig_q_meta[q_id]['q_type']
        row_str = f"{q_id:<14} {q_type:<14}"
        vals = []
        for m in MODELS:
            if m not in all_results:
                continue
            pct = all_results[m]['negation_accuracy'].get(q_id, {}).get('pct')
            row_str += f" {str(pct) if pct is not None else '—':>8}"
            if pct is not None:
                vals.append(pct)
        avg = round(sum(vals)/len(vals), 1) if vals else None
        row_str += f"  {str(avg) if avg is not None else '—':>6}"
        print(row_str)

    print("\n--- Model-level negation accuracy ---")
    for m in MODELS:
        if m not in all_results:
            continue
        vals = [v['pct'] for v in all_results[m]['negation_accuracy'].values()
                if v['pct'] is not None]
        avg = round(sum(vals)/len(vals), 1) if vals else None
        flag = '← near chance (negation failure)' if avg and avg < 55 else ''
        print(f"  {m:<14} {avg}%  {flag}")

    print(f"\nDone. Full results: {out_path}")

if __name__ == "__main__":
    main()
