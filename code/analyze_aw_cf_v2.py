"""
analyze_aw_cf_v2.py
-------------------
Scene-Change Sensitivity (SCS) analysis for Adverse Weather counterfactual pairs.

This version differs from analyze_aw_cf.py in ONE way:
  - Snowy CF data is sourced from the v2 CSVs (SD-augmented snowy) instead of
    the JarvisIR _cf.csv files. Per Shankar's feedback the JarvisIR snow output
    was visually too subtle; SD snowy from manifest v2 is used instead.
  - Fog and rain CF data continue to come from the JarvisIR _cf.csv files.

Data sources per condition:
  - Fog, Rain  : adverse_weather_<model>_cf.csv          (JarvisIR, Mode A)
                 adverse_weather_<model>_noimage_cf.csv  (JarvisIR, Mode C)
  - Snow       : adverse_weather_<model>_v2.csv          (SD, is_augmented=True, weather=snowy)
                 adverse_weather_<model>_noimage_v2.csv  (SD, is_augmented=True, weather=snowy)

Originals (clear Mode A) for pairing:
  - adverse_weather_<model>_v2.csv  (is_augmented=False)

SCS = % of (image, question) pairs where model flips its answer
      when weather condition is added (clear → fog/rain/snow)

Output:
  - aw_cf_analysis_v2.json
  - printed summary tables
"""

import csv
import json
import os
from collections import defaultdict

RESULTS_DIR = "/home2/muskan.singh/results"
OUTPUT_DIR  = "/home2/muskan.singh/results"

MODELS = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]

# ── helpers ──────────────────────────────────────────────────────────────────

def norm_ans(val):
    return str(val).strip().lower()

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def build_index(rows, key_fields):
    idx = {}
    for r in rows:
        key = tuple(r[f] for f in key_fields)
        idx[key] = r
    return idx

def filter_main_eval_yn(rows):
    """Keep only main_eval rows with clean yes/no extractions."""
    return [r for r in rows
            if r.get('use_for') == 'main_eval'
            and norm_ans(r.get('extracted_ans','')) in ('yes','no')]

def filter_cf_yn(rows):
    """Keep CF rows with clean yes/no. Accepts use_for=counterfactual OR main_eval."""
    return [r for r in rows
            if r.get('use_for') in ('main_eval', 'counterfactual')
            and norm_ans(r.get('extracted_ans','')) in ('yes','no')]

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for model in MODELS:
        orig_path   = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_v2.csv")
        cf_path     = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_cf.csv")
        cfc_path    = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_noimage_cf.csv")
        v2_noimg    = os.path.join(RESULTS_DIR, f"adverse_weather_{model}_noimage_v2.csv")

        missing = [p for p in [orig_path, cf_path, v2_noimg] if not os.path.exists(p)]
        if missing:
            print(f"[SKIP] {model}: missing files: {missing}")
            continue

        rows_orig_all = load_csv(orig_path)
        rows_cf_all   = load_csv(cf_path)
        rows_cfc_all  = load_csv(cfc_path) if os.path.exists(cfc_path) else []
        rows_v2_noimg_all = load_csv(v2_noimg)

        # ── Originals (clear Mode A): from v2, is_augmented=False ─────────────
        rows_orig = [r for r in rows_orig_all if r.get('is_augmented') == 'False']
        rows_orig = filter_main_eval_yn(rows_orig)

        # ── CF Mode A: JarvisIR fog + rain, SD snowy from v2 ──────────────────
        # JarvisIR: drop snowy
        rows_cf_jarvis = [r for r in rows_cf_all if r.get('weather') in ('foggy','rainy')]
        rows_cf_jarvis = filter_cf_yn(rows_cf_jarvis)

        # SD snowy from v2 (is_augmented=True, weather=snowy)
        rows_cf_sd_snow = [r for r in rows_orig_all
                           if r.get('is_augmented') == 'True'
                           and r.get('weather') == 'snowy']
        rows_cf_sd_snow = filter_cf_yn(rows_cf_sd_snow)

        rows_cf = rows_cf_jarvis + rows_cf_sd_snow

        # ── CF Mode C: JarvisIR fog + rain noimage, SD snowy noimage from v2 ──
        rows_cfc_jarvis = [r for r in rows_cfc_all if r.get('weather') in ('foggy','rainy')]
        rows_cfc_jarvis = filter_cf_yn(rows_cfc_jarvis)

        rows_cfc_sd_snow = [r for r in rows_v2_noimg_all
                            if r.get('is_augmented') == 'True'
                            and r.get('weather') == 'snowy']
        rows_cfc_sd_snow = filter_cf_yn(rows_cfc_sd_snow)

        rows_cfc = rows_cfc_jarvis + rows_cfc_sd_snow

        # Index originals by (source clear image, q_id)
        orig_idx = build_index(rows_orig, ['image_id', 'q_id'])
        # Index Mode C by (augmented image, q_id)
        cfc_idx  = build_index(rows_cfc,  ['image_id', 'q_id'])

        print(f"\n[{model}]")
        print(f"  originals (clear, main_eval): {len(rows_orig)}")
        print(f"  CF Mode A (JarvisIR fog+rain): {len(rows_cf_jarvis)}")
        print(f"  CF Mode A (SD snowy from v2):  {len(rows_cf_sd_snow)}")
        print(f"  CF Mode C (JarvisIR fog+rain): {len(rows_cfc_jarvis)}")
        print(f"  CF Mode C (SD snowy from v2):  {len(rows_cfc_sd_snow)}")

        # ── SCS computation ───────────────────────────────────────────────────
        scs_pairs = []

        for cf_row in rows_cf:
            src_img = cf_row.get('source_image_id', '')
            q_id    = cf_row['q_id']
            weather = cf_row['weather']
            aug_img = cf_row['image_id']

            orig_row = orig_idx.get((src_img, q_id))
            if orig_row is None:
                continue

            orig_ans = norm_ans(orig_row['extracted_ans'])
            cf_ans   = norm_ans(cf_row['extracted_ans'])
            orig_gt  = norm_ans(orig_row['ground_truth'])
            cf_gt    = norm_ans(cf_row['ground_truth'])

            flipped  = (orig_ans != cf_ans)

            orig_correct = (orig_ans == orig_gt)
            cf_correct   = (cf_ans   == cf_gt)

            if orig_correct and cf_correct:
                direction = 'both_correct'
            elif not orig_correct and not cf_correct:
                direction = 'both_wrong'
            elif orig_correct and not cf_correct:
                direction = 'correct_to_wrong'
            else:
                direction = 'wrong_to_correct'

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
        scs_overall = round(n_flipped / n * 100, 1)

        directions = {d: sum(1 for p in scs_pairs if p['direction']==d) for d in
                      ['both_correct','both_wrong','correct_to_wrong','wrong_to_correct']}

        # SCS per weather condition
        scs_by_weather = {}
        for w in ['foggy','rainy','snowy']:
            w_pairs = [p for p in scs_pairs if p['weather'] == w]
            if w_pairs:
                scs_by_weather[w] = {
                    'scs': round(sum(1 for p in w_pairs if p['flipped']) / len(w_pairs) * 100, 1),
                    'n'  : len(w_pairs),
                }

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
            'snow_source'   : 'SD (manifest v2)',
            'fog_rain_source': 'JarvisIR (_cf.csv)',
        }
        print(f"  ✅ done — {n} pairs, SCS={scs_overall}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "aw_cf_analysis_v2.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Print: Overall SCS ────────────────────────────────────────────────────
    print("\n" + "="*78)
    print("SCENE-CHANGE SENSITIVITY (SCS) — OVERALL")
    print("Snow: SD (manifest v2)  |  Fog/Rain: JarvisIR (_cf.csv)")
    print("="*78)
    print(f"\n{'Model':<14} {'SCS%':>6} {'→Wrong':>8} {'→Correct':>10} {'BothOK':>8} {'BothWrong':>10} {'ModeC_stab':>11}")
    print("-"*78)
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
              f"{str(r['cfc_stability']):>11}")

    # ── Print: SCS by weather ─────────────────────────────────────────────────
    print("\n" + "="*78)
    print("SCS BY WEATHER CONDITION")
    print("="*78)
    print(f"\n{'Model':<14} {'Foggy':>14} {'Rainy':>14} {'Snowy (SD)':>14}")
    print("-"*60)
    for model in MODELS:
        if model not in all_results:
            continue
        w = all_results[model]['scs_by_weather']
        def fmt(c):
            x = w.get(c)
            return f"{x['scs']}% (n={x['n']})" if x else '—'
        print(f"{model:<14} {fmt('foggy'):>14} {fmt('rainy'):>14} {fmt('snowy'):>14}")

    # ── Print: SCS per question ───────────────────────────────────────────────
    print("\n" + "="*78)
    print("SCS PER QUESTION (% flip rate)")
    print("="*78)
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
