"""
analyze_junctions.py
--------------------
Full cross-model analysis for Junctions & Intersections category.
Run AFTER extractor fix (use *_fixed.csv files).

Produces:
  - junctions_analysis.json  — full structured results
  - junctions_analysis.txt   — human-readable summary

Analysis axes:
  1. Overall accuracy per model (main_eval only, exclude unclear)
  2. GT=yes vs GT=no accuracy (bias metric)
  3. VGS per model (VGS_yes for Moondream/SmolVLM, VGS_overall for others)
  4. Per-question accuracy + VGS across all models
  5. Internal consistency: JN_ADV_01 vs JN_ADV_02 (green=proceed vs red=stop)
  6. Unclear rate per model (instruction-following metric)
  7. Failure mode flags per question
"""

import csv
import json
import os
from collections import defaultdict

RESULTS_DIR = "/home2/muskan.singh/results"
OUTPUT_DIR  = "/home2/muskan.singh/results"

MODELS = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]

# VGS method per model
# confounded = defaults to 'no' on blank image → use VGS_yes only
CONFOUNDED = {"moondream", "smolvlm"}

# Internal consistency pair
CONSISTENCY_PAIR = ("JN_ADV_01", "JN_ADV_02")  # green=yes(proceed) vs red/yellow=no(stop)

# ── helpers ──────────────────────────────────────────────────────────────────

def norm_correct(val):
    """Normalise True/False/TRUE/FALSE → bool."""
    return str(val).strip().lower() == "true"

def norm_ans(val):
    return str(val).strip().lower()

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def acc(rows):
    """Accuracy excluding unclear."""
    valid = [r for r in rows if norm_ans(r['extracted_ans']) in ('yes', 'no')]
    if not valid:
        return None, 0
    return sum(norm_correct(r['correct']) for r in valid) / len(valid), len(valid)

def vgs_yes(mode_a_rows, mode_c_rows):
    """
    VGS_yes = Mode A acc on GT=yes − Mode C acc on GT=yes
    Used for confounded models (Moondream, SmolVLM).
    """
    a_yes = [r for r in mode_a_rows
             if norm_ans(r['ground_truth']) == 'yes'
             and norm_ans(r['extracted_ans']) in ('yes','no')]
    c_yes = [r for r in mode_c_rows
             if norm_ans(r['ground_truth']) == 'yes'
             and norm_ans(r['extracted_ans']) in ('yes','no')]
    if not a_yes or not c_yes:
        return None
    a_acc = sum(norm_correct(r['correct']) for r in a_yes) / len(a_yes)
    c_acc = sum(norm_correct(r['correct']) for r in c_yes) / len(c_yes)
    return round((a_acc - c_acc) * 100, 1)

def vgs_overall(mode_a_rows, mode_c_rows):
    """
    VGS_overall = Mode A acc − Mode C acc
    Used for balanced models (PaliGemma, LLaVA-OV, InternVL3).
    """
    a_acc, a_n = acc(mode_a_rows)
    c_acc, c_n = acc(mode_c_rows)
    if a_acc is None or c_acc is None:
        return None
    return round((a_acc - c_acc) * 100, 1)

def per_question_stats(rows):
    """Returns dict: q_id → {acc, gt_yes_acc, gt_no_acc, n_valid, n_unclear}"""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r['q_id']].append(r)
    stats = {}
    for q_id, qrows in by_q.items():
        valid   = [r for r in qrows if norm_ans(r['extracted_ans']) in ('yes','no')]
        unclear = [r for r in qrows if norm_ans(r['extracted_ans']) == 'unclear']
        yes_rows = [r for r in valid if norm_ans(r['ground_truth']) == 'yes']
        no_rows  = [r for r in valid if norm_ans(r['ground_truth']) == 'no']
        stats[q_id] = {
            'n_total'   : len(qrows),
            'n_valid'   : len(valid),
            'n_unclear' : len(unclear),
            'acc'       : round(sum(norm_correct(r['correct']) for r in valid) / len(valid) * 100, 1) if valid else None,
            'gt_yes_acc': round(sum(norm_correct(r['correct']) for r in yes_rows) / len(yes_rows) * 100, 1) if yes_rows else None,
            'gt_no_acc' : round(sum(norm_correct(r['correct']) for r in no_rows)  / len(no_rows)  * 100, 1) if no_rows  else None,
            'use_for'   : qrows[0].get('use_for', ''),
            'q_type'    : qrows[0].get('q_type', ''),
            'question'  : qrows[0].get('question', ''),
        }
    return stats

def internal_consistency(mode_a_rows, q1, q2):
    """
    For each image, check if model answered q1 and q2 consistently.
    q1 = JN_ADV_01 (green light → should proceed → GT=yes)
    q2 = JN_ADV_02 (red/yellow light → should stop  → GT=yes when stopped)
    Consistent = both correct OR both incorrect (not one right one wrong).
    Returns: {consistent%, both_correct%, both_wrong%, inconsistent%, n_images}
    """
    # group by image
    by_image = defaultdict(dict)
    for r in mode_a_rows:
        if r['q_id'] in (q1, q2) and norm_ans(r['extracted_ans']) in ('yes','no'):
            by_image[r['image_id']][r['q_id']] = norm_correct(r['correct'])

    both_correct = both_wrong = inconsistent = 0
    n = 0
    for image_id, qs in by_image.items():
        if q1 in qs and q2 in qs:
            n += 1
            c1, c2 = qs[q1], qs[q2]
            if c1 and c2:
                both_correct += 1
            elif not c1 and not c2:
                both_wrong += 1
            else:
                inconsistent += 1

    if n == 0:
        return None
    return {
        'n_images'      : n,
        'both_correct'  : round(both_correct / n * 100, 1),
        'both_wrong'    : round(both_wrong   / n * 100, 1),
        'inconsistent'  : round(inconsistent / n * 100, 1),
        'consistent_pct': round((both_correct + both_wrong) / n * 100, 1),
    }

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    results = {}

    for model in MODELS:
        a_path = os.path.join(RESULTS_DIR, f"junctions_{model}_ji_v1_fixed.csv")
        c_path = os.path.join(RESULTS_DIR, f"junctions_{model}_noimage_ji_v1.csv")

        if not os.path.exists(a_path):
            print(f"[SKIP] {a_path} not found")
            continue
        if not os.path.exists(c_path):
            print(f"[WARN] Mode C not found for {model} — VGS will be None")

        rows_a = load_csv(a_path)
        rows_c = load_csv(c_path) if os.path.exists(c_path) else []

        # Filter to main_eval only for accuracy metrics
        main_a = [r for r in rows_a if r.get('use_for') == 'main_eval']
        main_c = [r for r in rows_c if r.get('use_for') == 'main_eval']

        # --- Overall accuracy ---
        overall_acc, n_valid = acc(main_a)
        n_unclear = sum(1 for r in main_a if norm_ans(r['extracted_ans']) == 'unclear')
        n_total   = len(main_a)

        # --- GT=yes / GT=no accuracy ---
        yes_rows = [r for r in main_a
                    if norm_ans(r['ground_truth']) == 'yes'
                    and norm_ans(r['extracted_ans']) in ('yes','no')]
        no_rows  = [r for r in main_a
                    if norm_ans(r['ground_truth']) == 'no'
                    and norm_ans(r['extracted_ans']) in ('yes','no')]

        gt_yes_acc = sum(norm_correct(r['correct']) for r in yes_rows) / len(yes_rows) if yes_rows else None
        gt_no_acc  = sum(norm_correct(r['correct']) for r in no_rows)  / len(no_rows)  if no_rows  else None

        bias = None
        if gt_yes_acc is not None and gt_no_acc is not None:
            gap = gt_yes_acc - gt_no_acc
            if gap > 0.15:
                bias = "affirmation (over-yes)"
            elif gap < -0.15:
                bias = "negation (over-no)"
            else:
                bias = "balanced"

        # --- VGS ---
        if model in CONFOUNDED:
            vgs = vgs_yes(main_a, main_c)
            vgs_method = "VGS_yes"
        else:
            vgs = vgs_overall(main_a, main_c)
            vgs_method = "VGS_overall"

        # --- Per-question stats ---
        pq = per_question_stats(main_a)

        # VGS per question (using same method as model-level)
        pq_c = per_question_stats(main_c) if main_c else {}
        for q_id in pq:
            qa_rows = [r for r in main_a if r['q_id'] == q_id]
            qc_rows = [r for r in main_c if r['q_id'] == q_id] if main_c else []
            if model in CONFOUNDED:
                pq[q_id]['vgs'] = vgs_yes(qa_rows, qc_rows)
            else:
                pq[q_id]['vgs'] = vgs_overall(qa_rows, qc_rows)

        # --- Internal consistency ---
        consistency = internal_consistency(rows_a, *CONSISTENCY_PAIR)

        results[model] = {
            'overall_acc'   : round(overall_acc * 100, 1) if overall_acc is not None else None,
            'gt_yes_acc'    : round(gt_yes_acc * 100, 1)  if gt_yes_acc  is not None else None,
            'gt_no_acc'     : round(gt_no_acc  * 100, 1)  if gt_no_acc   is not None else None,
            'bias'          : bias,
            'vgs'           : vgs,
            'vgs_method'    : vgs_method,
            'n_total'       : n_total,
            'n_valid'       : n_valid,
            'n_unclear'     : n_unclear,
            'unclear_pct'   : round(n_unclear / n_total * 100, 1) if n_total else None,
            'per_question'  : pq,
            'consistency'   : consistency,
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "junctions_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("JUNCTIONS & INTERSECTIONS — CROSS-MODEL SUMMARY")
    print("="*70)

    print(f"\n{'Model':<14} {'Acc%':>6} {'GT=yes%':>8} {'GT=no%':>8} {'VGS':>8} {'Method':<14} {'Unclear%':>9} {'Bias'}")
    print("-"*80)
    for model in MODELS:
        if model not in results:
            continue
        r = results[model]
        print(f"{model:<14} {str(r['overall_acc']):>6} {str(r['gt_yes_acc']):>8} "
              f"{str(r['gt_no_acc']):>8} {str(r['vgs']):>8} {r['vgs_method']:<14} "
              f"{str(r['unclear_pct']):>9} {r['bias']}")

    # Per-question table (main_eval only, sorted by avg VGS descending)
    print("\n" + "="*70)
    print("PER-QUESTION ACCURACY (all models)")
    print("="*70)

    # Collect all q_ids from first available model
    first = next(m for m in MODELS if m in results)
    q_ids = sorted(results[first]['per_question'].keys())

    header = f"{'Q_ID':<14} {'Type':<8}"
    for m in MODELS:
        if m in results:
            header += f" {m[:7]:>9}"
    header += f" {'VGS_avg':>8}"
    print(header)
    print("-" * len(header))

    for q_id in q_ids:
        q_type = results[first]['per_question'][q_id]['q_type']
        row_str = f"{q_id:<14} {q_type:<8}"
        vgs_vals = []
        for m in MODELS:
            if m not in results:
                continue
            pq = results[m]['per_question'].get(q_id)
            a = str(pq['acc']) if pq and pq['acc'] is not None else "—"
            row_str += f" {a:>9}"
            if pq and pq['vgs'] is not None:
                vgs_vals.append(pq['vgs'])
        avg_vgs = round(sum(vgs_vals)/len(vgs_vals), 1) if vgs_vals else None
        row_str += f" {str(avg_vgs):>8}"
        print(row_str)

    # GT=no accuracy per question (bias detection)
    print("\n" + "="*70)
    print("GT=NO ACCURACY PER QUESTION (affirmation bias detector)")
    print("="*70)
    header2 = f"{'Q_ID':<14}"
    for m in MODELS:
        if m in results:
            header2 += f" {m[:7]:>9}"
    print(header2)
    print("-" * len(header2))
    for q_id in q_ids:
        row_str = f"{q_id:<14}"
        for m in MODELS:
            if m not in results:
                continue
            pq = results[m]['per_question'].get(q_id)
            v = str(pq['gt_no_acc']) if pq and pq['gt_no_acc'] is not None else "—"
            row_str += f" {v:>9}"
        print(row_str)

    # Internal consistency
    print("\n" + "="*70)
    print(f"INTERNAL CONSISTENCY: {CONSISTENCY_PAIR[0]} vs {CONSISTENCY_PAIR[1]}")
    print("(Green=proceed vs Red/Yellow=stop — model must give opposite answers)")
    print("="*70)
    for model in MODELS:
        if model not in results:
            continue
        c = results[model]['consistency']
        if c:
            print(f"{model:<14} consistent={c['consistent_pct']}%  "
                  f"both_correct={c['both_correct']}%  "
                  f"both_wrong={c['both_wrong']}%  "
                  f"inconsistent={c['inconsistent']}%  "
                  f"(n={c['n_images']})")
        else:
            print(f"{model:<14} — no data")

    print(f"\nDone. Full results: {json_path}")

if __name__ == "__main__":
    main()
