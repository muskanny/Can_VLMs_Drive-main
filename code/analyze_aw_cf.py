"""
analyze_aw_cf.py
----------------
Scene-Change Sensitivity (SCS) analysis for Adverse Weather counterfactual pairs.
Compares model answers on clear original images vs JarvisIR augmented versions.

SCS = % of (image, question) pairs where model flips its answer
      when weather condition is added (clear → fog/rain/snow)

Requires:
  - adverse_weather_<model>_v2.csv     (original images — Mode A)
  - adverse_weather_<model>_cf.csv     (augmented images — Mode A)
  - adverse_weather_<model>_noimage_cf.csv (augmented Mode C — for language prior check)

Output:
  - aw_cf_analysis.json
  - printed summary tables

Analysis axes:
  1. Overall SCS per model (% answer flips on augmented vs original)
  2. SCS per question per model
  3. SCS per weather condition (fog vs rain vs snow)
  4. Direction of flip: correct→wrong vs wrong→correct vs correct→correct vs wrong→wrong
  5. Mode C stability on augmented images (do models answer differently without image?)
"""

import csv
import json
import os
from collections import defaultdict

RESULTS_DIR = "/home2/muskan.singh/results"
OUTPUT_DIR  = "/home2/muskan.singh/results"

MODELS = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]

# Only analyse main_eval questions
# Only questions where GT actually changes with weather augmentation are meaningful
# (GT flip rate from manifest analysis: ~51% overall)

# ── helpers ──────────────────────────────────────────────────────────────────

def norm_correct(val):
    return str(val).strip().lower() == "true"

def norm_ans(val):
    return str(val).strip().lower()

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def build_index(rows, key_fields):
    """Index rows by tuple of key fields."""
    idx = {}
    for r in rows:
        key = tuple(r[f] for f in key_fields)
        idx[key] = r
    return idx

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for model in MODELS:
        orig_path = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_v2.csv")
        cf_path   = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_cf.csv")
        cfc_path  = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_noimage_cf.csv")

        for p in [orig_path, cf_path]:
            if not os.path.exists(p):
                print(f"[SKIP] {p} not found")
                continue

        rows_orig = load_csv(orig_path)
        rows_cf   = load_csv(cf_path)
        rows_cfc  = load_csv(cfc_path) if os.path.exists(cfc_path) else []

        # Filter to main_eval only
        rows_orig = [r for r in rows_orig if r['use_for'] == 'main_eval'
                     and norm_ans(r['extracted_ans']) in ('yes','no')
                     and r.get('is_augmented','False') == 'False']
        rows_cf   = [r for r in rows_cf   if r['use_for'] == 'main_eval'
                     and norm_ans(r['extracted_ans']) in ('yes','no')]
        rows_cfc  = [r for r in rows_cfc  if r['use_for'] == 'main_eval'
                     and norm_ans(r['extracted_ans']) in ('yes','no')]

        # Index originals by (image_id, q_id)
        orig_idx = build_index(rows_orig, ['image_id', 'q_id'])
        # Index CF Mode C by (image_id, q_id)
        cfc_idx  = build_index(rows_cfc,  ['image_id', 'q_id'])

        # ── SCS computation ───────────────────────────────────────────────────
        # For each CF row: find matching original (source_image_id, q_id)
        scs_pairs = []

        for cf_row in rows_cf:
            src_img  = cf_row.get('source_image_id', '')
            q_id     = cf_row['q_id']
            weather  = cf_row['weather']
            aug_img  = cf_row['image_id']

            orig_row = orig_idx.get((src_img, q_id))
            if orig_row is None:
                continue

            orig_ans = norm_ans(orig_row['extracted_ans'])
            cf_ans   = norm_ans(cf_row['extracted_ans'])
            orig_gt  = norm_ans(orig_row['ground_truth'])
            cf_gt    = norm_ans(cf_row['ground_truth'])

            flipped  = (orig_ans != cf_ans)

            # Direction of flip
            orig_correct = (orig_ans == orig_gt)
            cf_correct   = (cf_ans   == cf_gt)

            if orig_correct and cf_correct:
                direction = 'both_correct'
            elif not orig_correct and not cf_correct:
                direction = 'both_wrong'
            elif orig_correct and not cf_correct:
                direction = 'correct_to_wrong'   # model degraded
            else:
                direction = 'wrong_to_correct'   # model improved

            # Mode C answer on augmented image
            cfc_row = cfc_idx.get((aug_img, q_id))
            cfc_ans = norm_ans(cfc_row['extracted_ans']) if cfc_row else None

            scs_pairs.append({
                'source_image_id': src_img,
                'aug_image_id'   : aug_img,
                'q_id'           : q_id,
                'q_type'         : cf_row.get('q_type',''),
                'weather'        : weather,
                'orig_ans'       : orig_ans,
                'cf_ans'         : cf_ans,
                'orig_gt'        : orig_gt,
                'cf_gt'          : cf_gt,
                'flipped'        : flipped,
                'direction'      : direction,
                'cfc_ans'        : cfc_ans,
            })

        if not scs_pairs:
            print(f"[WARN] No SCS pairs for {model}")
            continue

        n = len(scs_pairs)
        n_flipped = sum(1 for p in scs_pairs if p['flipped'])

        # Overall SCS
        scs_overall = round(n_flipped / n * 100, 1)

        # Direction breakdown
        directions = {d: sum(1 for p in scs_pairs if p['direction']==d) for d in
                      ['both_correct','both_wrong','correct_to_wrong','wrong_to_correct']}

        # SCS per weather condition
        scs_by_weather = {}
        for w in ['foggy','rainy','snowy']:
            w_pairs = [p for p in scs_pairs if p['weather'] == w]
            if w_pairs:
                scs_by_weather[w] = round(sum(1 for p in w_pairs if p['flipped']) / len(w_pairs) * 100, 1)

        # SCS per question
        scs_by_q = {}
        q_ids = sorted(set(p['q_id'] for p in scs_pairs))
        for q_id in q_ids:
            q_pairs = [p for p in scs_pairs if p['q_id'] == q_id]
            if q_pairs:
                scs_by_q[q_id] = {
                    'scs'      : round(sum(1 for p in q_pairs if p['flipped']) / len(q_pairs) * 100, 1),
                    'n'        : len(q_pairs),
                    'q_type'   : q_pairs[0]['q_type'],
                    'by_weather': {
                        w: round(sum(1 for p in q_pairs if p['flipped'] and p['weather']==w) /
                                 max(1, sum(1 for p in q_pairs if p['weather']==w)) * 100, 1)
                        for w in ['foggy','rainy','snowy']
                    }
                }

        # Mode C stability on augmented images
        # Does model give same answer with/without image on augmented versions?
        cfc_pairs = [(p['cf_ans'], p['cfc_ans']) for p in scs_pairs if p['cfc_ans'] is not None]
        cfc_same  = sum(1 for a,b in cfc_pairs if a == b)
        cfc_stability = round(cfc_same / len(cfc_pairs) * 100, 1) if cfc_pairs else None

        all_results[model] = {
            'n_pairs'       : n,
            'n_flipped'     : n_flipped,
            'scs_overall'   : scs_overall,
            'directions'    : directions,
            'scs_by_weather': scs_by_weather,
            'scs_by_q'      : scs_by_q,
            'cfc_stability' : cfc_stability,
        }
        print(f"  {model} done — {n} pairs, SCS={scs_overall}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "aw_cf_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Print: Overall SCS ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("SCENE-CHANGE SENSITIVITY (SCS) — OVERALL")
    print("% of pairs where model flips answer: clear original → augmented weather")
    print("="*72)
    print(f"\n{'Model':<14} {'SCS%':>6} {'→Wrong':>8} {'→Correct':>10} {'BothOK':>8} {'BothWrong':>10} {'ModeC_stable':>13}")
    print("-"*72)
    for model in MODELS:
        if model not in all_results:
            continue
        r = all_results[model]
        d = r['directions']
        n = r['n_pairs']
        print(f"{model:<14} {r['scs_overall']:>6} "
              f"{round(d['correct_to_wrong']/n*100,1):>8} "
              f"{round(d['wrong_to_correct']/n*100,1):>10} "
              f"{round(d['both_correct']/n*100,1):>8} "
              f"{round(d['both_wrong']/n*100,1):>10} "
              f"{str(r['cfc_stability']):>13}")

    # ── Print: SCS by weather ─────────────────────────────────────────────────
    print("\n" + "="*72)
    print("SCS BY WEATHER CONDITION")
    print("="*72)
    print(f"\n{'Model':<14} {'Foggy':>8} {'Rainy':>8} {'Snowy':>8}")
    print("-"*44)
    for model in MODELS:
        if model not in all_results:
            continue
        w = all_results[model]['scs_by_weather']
        print(f"{model:<14} {str(w.get('foggy','—')):>8} {str(w.get('rainy','—')):>8} {str(w.get('snowy','—')):>8}")

    # ── Print: SCS per question ───────────────────────────────────────────────
    print("\n" + "="*72)
    print("SCS PER QUESTION (% flip rate across all models)")
    print("="*72)
    first = next(m for m in MODELS if m in all_results)
    q_ids = sorted(all_results[first]['scs_by_q'].keys())

    header = f"{'Q_ID':<14} {'Type':<12}"
    for m in MODELS:
        if m in all_results:
            header += f" {m[:7]:>8}"
    header += f"  {'Avg':>6}"
    print(header)
    print("-" * len(header))

    for q_id in q_ids:
        q_type = all_results[first]['scs_by_q'].get(q_id,{}).get('q_type','')
        row_str = f"{q_id:<14} {q_type:<12}"
        vals = []
        for m in MODELS:
            if m not in all_results:
                continue
            pct = all_results[m]['scs_by_q'].get(q_id, {}).get('scs')
            row_str += f" {str(pct) if pct is not None else '—':>8}"
            if pct is not None:
                vals.append(pct)
        avg = round(sum(vals)/len(vals), 1) if vals else None
        row_str += f"  {str(avg):>6}"
        print(row_str)

    print(f"\nDone. Full results: {out_path}")

if __name__ == "__main__":
    main()
