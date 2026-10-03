"""
extract_aw_tables.py
--------------------
Extracts all AW analysis tables in TSV format for Google Sheets paste.
Covers:
  Table 1: Model-level summary (Axis 1)
  Table 2: Per-question accuracy across models (Axis 2)
  Table 3: Mode A vs Mode C — visual grounding (Axis 3)
  Table 4: Linguistic robustness — paraphrase + negation (Axis 4)
  Table 5: Counterfactual SCS by weather (Axis 5)
  Table 6: Per-weather accuracy breakdown

Uses: adverse_weather_<model>_v2.csv (main analysis)
      adverse_weather_<model>_noimage_v2.csv (Mode C)
      aw_cf_analysis.json (SCS)
      aw_linguistic_analysis.json (linguistic)
"""

import csv, json, os
from collections import defaultdict

RESULTS = '/home2/muskan.singh/results'
MODELS  = ['moondream', 'paligemma', 'smolvlm', 'llava_ov', 'internvl3']
MODEL_LABELS = {
    'moondream': 'Moondream 2B',
    'paligemma': 'PaliGemma 3B',
    'smolvlm'  : 'SmolVLM 2.2B',
    'llava_ov' : 'LLaVA-OV 8B',
    'internvl3': 'InternVL3 8B',
}

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def pct(num, den):
    return round(num/den*100, 1) if den else 0.0

def tsv(*cols):
    return '\t'.join(str(c) for c in cols)

# ── Load all data ─────────────────────────────────────────────
data     = {}   # model → rows (main_eval, is_augmented=False)
data_all = {}   # model → all rows
modec    = {}   # model → mode C rows

for m in MODELS:
    path_a = os.path.join(RESULTS, f'adverse_weather_{m}_v2.csv')
    path_c = os.path.join(RESULTS, f'adverse_weather_{m}_noimage_v2.csv')
    if not os.path.exists(path_a):
        print(f'MISSING: {path_a}')
        continue
    all_rows = load_csv(path_a)
    data_all[m] = all_rows
    data[m] = [r for r in all_rows
               if r['use_for'] == 'main_eval'
               and r['is_augmented'] == 'False']
    if os.path.exists(path_c):
        mc = load_csv(path_c)
        modec[m] = [r for r in mc
                    if r['use_for'] == 'main_eval'
                    and r['is_augmented'] == 'False']

# Load analysis JSONs
with open(os.path.join(RESULTS, 'aw_linguistic_analysis.json')) as f:
    ling_data = json.load(f)
with open(os.path.join(RESULTS, 'aw_cf_analysis.json')) as f:
    cf_data = json.load(f)

# ── Get all q_ids in order ────────────────────────────────────
q_info = {}  # q_id → {question, q_type}
for m in MODELS:
    if m not in data: continue
    for r in data[m]:
        if r['q_id'] not in q_info:
            q_info[r['q_id']] = {
                'question': r['question'],
                'q_type'  : r['q_type'],
                'gt'      : r['ground_truth'],
            }
q_ids = sorted(q_info.keys())

# ── TABLE 1: Model-level summary ──────────────────────────────
print('\n' + '='*80)
print('TABLE 1: Model-Level Summary (AW — main_eval, original images only)')
print('='*80)
print(tsv('Model', 'Overall Acc%', 'GT=yes Acc%', 'GT=no Acc%',
          'Affirmation Gap', 'Unclear%', 'Avg Response Time(s)'))

for m in MODELS:
    if m not in data: continue
    rows = data[m]
    total = len(rows)
    correct = sum(1 for r in rows if r['correct'].strip().lower() == 'true')
    yes_rows = [r for r in rows if r['ground_truth'].strip().lower() == 'yes']
    no_rows  = [r for r in rows if r['ground_truth'].strip().lower() == 'no']
    yes_correct = sum(1 for r in yes_rows if r['correct'].strip().lower() == 'true')
    no_correct  = sum(1 for r in no_rows  if r['correct'].strip().lower() == 'true')
    unclear = sum(1 for r in rows if r['extracted_ans'].strip().lower() == 'unclear')
    avg_time = sum(float(r['response_time']) for r in rows if r['response_time']) / total

    overall   = pct(correct, total)
    yes_acc   = pct(yes_correct, len(yes_rows))
    no_acc    = pct(no_correct,  len(no_rows))
    aff_gap   = round(yes_acc - no_acc, 1)
    unc_pct   = pct(unclear, total)

    print(tsv(MODEL_LABELS[m], overall, yes_acc, no_acc,
              aff_gap, unc_pct, round(avg_time, 2)))

# ── TABLE 2: Per-question accuracy across models ──────────────
print('\n' + '='*80)
print('TABLE 2: Per-Question Accuracy % (main_eval, original images)')
print('='*80)
header = ['Q_ID', 'Q_Type', 'GT', 'Question (truncated)'] + \
         [MODEL_LABELS[m] for m in MODELS if m in data] + ['Mean']
print(tsv(*header))

# Index: model → q_id → {correct, total}
q_model_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for m in MODELS:
    if m not in data: continue
    for r in data[m]:
        qid = r['q_id']
        q_model_stats[m][qid][1] += 1
        if r['correct'].strip().lower() == 'true':
            q_model_stats[m][qid][0] += 1

for qid in q_ids:
    info = q_info[qid]
    vals = []
    for m in MODELS:
        if m not in data: continue
        c, t = q_model_stats[m][qid]
        vals.append(pct(c, t))
    mean = round(sum(vals)/len(vals), 1) if vals else 0
    row = [qid, info['q_type'], info['gt'], info['question'][:60]] + vals + [mean]
    print(tsv(*row))

# ── TABLE 3: Mode A vs Mode C (Visual Grounding) ──────────────
print('\n' + '='*80)
print('TABLE 3: Mode A vs Mode C — Visual Grounding')
print('='*80)
print(tsv('Model', 'Mode A Acc%', 'Mode C Acc%', 'Mode C GT=yes Acc%',
          'Mode C GT=no Acc%', 'VGS (Mode A - Mode C)', 'Mode C Default'))

for m in MODELS:
    if m not in data or m not in modec: continue
    a_rows = data[m]
    c_rows = modec[m]

    a_total   = len(a_rows)
    a_correct = sum(1 for r in a_rows if r['correct'].strip().lower() == 'true')
    a_acc     = pct(a_correct, a_total)

    c_total   = len(c_rows)
    c_correct = sum(1 for r in c_rows if r['correct'].strip().lower() == 'true')
    c_acc     = pct(c_correct, c_total)

    c_yes = [r for r in c_rows if r['ground_truth'].strip().lower() == 'yes']
    c_no  = [r for r in c_rows if r['ground_truth'].strip().lower() == 'no']
    c_yes_acc = pct(sum(1 for r in c_yes if r['correct'].strip().lower() == 'true'), len(c_yes))
    c_no_acc  = pct(sum(1 for r in c_no  if r['correct'].strip().lower() == 'true'), len(c_no))

    vgs = round(a_acc - c_acc, 1)

    # Mode C default
    c_ans = defaultdict(int)
    for r in c_rows:
        c_ans[r['extracted_ans'].strip().lower()] += 1
    default = max(c_ans, key=c_ans.get)
    default_pct = pct(c_ans[default], c_total)
    default_str = default + ' (' + str(default_pct) + '%)'

    print(tsv(MODEL_LABELS[m], a_acc, c_acc, c_yes_acc, c_no_acc, vgs, default_str))

# ── TABLE 4: Linguistic Robustness ────────────────────────────
print('\n' + '='*80)
print('TABLE 4: Linguistic Robustness — Paraphrase Consistency & Negation Accuracy')
print('='*80)
print(tsv('Model', 'Paraphrase Consistency%', 'Negation Accuracy%', 'Negation vs Chance'))

for m in MODELS:
    if m not in ling_data: continue
    para_vals = [v['pct'] for v in ling_data[m]['paraphrase_consistency'].values()
                 if v['pct'] is not None]
    neg_vals  = [v['pct'] for v in ling_data[m]['negation_accuracy'].values()
                 if v['pct'] is not None]
    para_avg = round(sum(para_vals)/len(para_vals), 1) if para_vals else 0
    neg_avg  = round(sum(neg_vals)/len(neg_vals),  1) if neg_vals  else 0
    vs_chance = round(neg_avg - 50.0, 1)
    flag = '← below chance' if neg_avg < 50 else ('← near chance' if neg_avg < 55 else '')
    print(tsv(MODEL_LABELS[m], para_avg, neg_avg, str(vs_chance) + '%  ' + flag))

# Per-question linguistic
print()
print(tsv('Q_ID', 'Q_Type') +
      '\t' + '\t'.join(MODEL_LABELS[m]+' Para%' for m in MODELS if m in ling_data) +
      '\t' + '\t'.join(MODEL_LABELS[m]+' Neg%'  for m in MODELS if m in ling_data))
for qid in q_ids:
    row = [qid, q_info[qid]['q_type']]
    for m in MODELS:
        if m not in ling_data: continue
        v = ling_data[m]['paraphrase_consistency'].get(qid, {})
        row.append(v.get('pct', '—'))
    for m in MODELS:
        if m not in ling_data: continue
        v = ling_data[m]['negation_accuracy'].get(qid, {})
        row.append(v.get('pct', '—'))
    print(tsv(*row))

# ── TABLE 5: Counterfactual SCS by weather ────────────────────
print('\n' + '='*80)
print('TABLE 5: Scene-Change Sensitivity (SCS) by Weather Condition')
print('Note: Snowy uses SD augmentation (JarvisIR snow too subtle)')
print('='*80)
print(tsv('Model', 'Fog SCS%', 'Rain SCS%', 'Snow SCS% (SD)',
          'Overall SCS%', 'Correct→Wrong', 'Wrong→Correct'))

for m in MODELS:
    if m not in cf_data: continue
    d = cf_data[m]
    fog   = d.get('scs_by_weather', {}).get('foggy', '—')
    rain  = d.get('scs_by_weather', {}).get('rainy', '—')
    snow  = d.get('scs_by_weather', {}).get('snowy', '—')
    overall = d.get('scs_overall', '—')
    dirs = d.get('directions', {})
    c2w  = dirs.get('correct_to_wrong', '—')
    w2c  = dirs.get('wrong_to_correct', '—')
    print(tsv(MODEL_LABELS[m], fog, rain, snow, overall, c2w, w2c))

# ── TABLE 6: Per-weather accuracy ────────────────────────────
print('\n' + '='*80)
print('TABLE 6: Per-Weather Accuracy % (main_eval, original images)')
print('='*80)
WEATHERS = ['rainy', 'foggy', 'snowy', 'night', 'clear']
print(tsv('Model', *[w.capitalize() for w in WEATHERS]))

for m in MODELS:
    if m not in data: continue
    row = [MODEL_LABELS[m]]
    for w in WEATHERS:
        w_rows = [r for r in data[m] if r['weather'] == w]
        w_correct = sum(1 for r in w_rows if r['correct'].strip().lower() == 'true')
        row.append(pct(w_correct, len(w_rows)))
    print(tsv(*row))

print('\nDone. Copy each table section into a separate Google Sheet tab.')
