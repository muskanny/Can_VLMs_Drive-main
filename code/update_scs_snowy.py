"""
update_scs_snowy.py
-------------------
Replaces JarvisIR snowy results in aw_cf_analysis.json with
SD snowy results from v2 CSVs (is_augmented=True, weather=snowy).
Per Shankar feedback: JarvisIR snow too subtle, use SD snowy instead.
"""
import csv, json, os
from collections import defaultdict

RESULTS_DIR  = "/home2/muskan.singh/results"
MODELS       = ["moondream", "paligemma", "smolvlm", "llava_ov", "internvl3"]

def load_sd_snowy(model):
    """Load SD snowy rows from v2 CSV."""
    path = f"{RESULTS_DIR}/adverse_weather_{model}_v2.csv"
    rows = list(csv.DictReader(open(path)))
    return [r for r in rows
            if r.get('weather','') == 'snowy'
            and r.get('is_augmented','') == 'True']

def load_clear_answers(model):
    """Load clear baseline answers from v2 CSV for source images."""
    path = f"{RESULTS_DIR}/adverse_weather_{model}_v2.csv"
    rows = list(csv.DictReader(open(path)))
    # index by (source_image_id, q_id) for clear non-augmented images
    idx = {}
    for r in rows:
        if r.get('is_augmented','') == 'False' and r.get('weather','') == 'clear':
            idx[(r['image_id'], r['q_id'])] = r['extracted_ans'].strip().lower()
    return idx

def compute_scs_for_model(model):
    snowy_rows  = load_sd_snowy(model)
    clear_index = load_clear_answers(model)

    # SCS = did answer flip between clear source and snowy augmented?
    by_q   = defaultdict(lambda: {'flipped': 0, 'total': 0})
    overall = {'flipped': 0, 'total': 0,
               'both_correct': 0, 'both_wrong': 0,
               'correct_to_wrong': 0, 'wrong_to_correct': 0}

    for r in snowy_rows:
        src_img = r.get('source_image_id', '')
        q_id    = r['q_id']
        aug_ans = r['extracted_ans'].strip().lower()
        gt      = r['ground_truth'].strip().lower()

        clear_ans = clear_index.get((src_img, q_id))
        if clear_ans is None or aug_ans not in ('yes','no') or clear_ans not in ('yes','no'):
            continue

        flipped = (clear_ans != aug_ans)
        by_q[q_id]['total']   += 1
        by_q[q_id]['flipped'] += int(flipped)
        overall['total']      += 1
        overall['flipped']    += int(flipped)

        # Direction analysis
        clear_correct = (clear_ans == gt)
        aug_correct   = (aug_ans   == gt)
        if clear_correct and aug_correct:     overall['both_correct']     += 1
        elif not clear_correct and not aug_correct: overall['both_wrong']  += 1
        elif clear_correct and not aug_correct:    overall['correct_to_wrong'] += 1
        else:                                      overall['wrong_to_correct'] += 1

    scs_overall = round(overall['flipped'] / overall['total'] * 100, 1) if overall['total'] else 0
    scs_by_q    = {q: round(v['flipped']/v['total']*100, 1) if v['total'] else 0
                   for q, v in by_q.items()}

    return {
        'n_pairs'   : overall['total'],
        'n_flipped' : overall['flipped'],
        'scs_overall': scs_overall,
        'snowy_source': 'sd_v2',  # flag to show this uses SD not JarvisIR
        'directions': {k: overall[k] for k in ['both_correct','both_wrong',
                                                'correct_to_wrong','wrong_to_correct']},
        'scs_by_q'  : scs_by_q,
    }

# Load existing analysis
analysis_path = f"{RESULTS_DIR}/aw_cf_analysis.json"
with open(analysis_path) as f:
    analysis = json.load(f)

print("Updating snowy SCS with SD results...")
for model in MODELS:
    v2_path = f"{RESULTS_DIR}/adverse_weather_{model}_v2.csv"
    if not os.path.exists(v2_path):
        print(f"  [SKIP] {model} — v2 CSV not found")
        continue

    snowy_scs = compute_scs_for_model(model)

    # Update only the snowy condition in existing analysis
    if model in analysis:
        analysis[model]['scs_by_weather']['snowy'] = snowy_scs['scs_overall']
        analysis[model]['snowy_sd_detail'] = snowy_scs
        print(f"  {model}: snowy SCS updated → {snowy_scs['scs_overall']}% "
              f"({snowy_scs['n_flipped']}/{snowy_scs['n_pairs']} flips)")
    else:
        print(f"  [WARN] {model} not in existing analysis")

# Save updated analysis
out_path = f"{RESULTS_DIR}/aw_cf_analysis_updated.json"
with open(out_path, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nSaved: {out_path}")

# Print comparison
print("\n=== SCS by weather — before vs after snowy swap ===")
print(f"{'Model':<14} {'Foggy':>8} {'Rainy':>8} {'Snowy (JarvisIR)':>18} {'Snowy (SD)':>12}")
print("-" * 64)
for model in MODELS:
    if model not in analysis:
        continue
    fog   = analysis[model].get('scs_by_weather',{}).get('foggy','—')
    rain  = analysis[model].get('scs_by_weather',{}).get('rainy','—')
    jarvis_snow = 7.7 if model == 'moondream' else '—'  # from old analysis
    sd_snow = analysis[model].get('snowy_sd_detail',{}).get('scs_overall','—')
    print(f"{model:<14} {str(fog):>8} {str(rain):>8} {str(jarvis_snow):>18} {str(sd_snow):>12}")
