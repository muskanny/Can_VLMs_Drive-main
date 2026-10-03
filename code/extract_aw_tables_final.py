"""
extract_aw_tables_final.py
--------------------------
Produces ALL Adverse Weather analysis tables in TSV format
for direct paste into Google Sheets.

Tables produced:
  T1  — Model-level summary (accuracy, bias, unclear rate)
  T2  — Per-question accuracy across all 5 models
  T3  — Per-weather accuracy breakdown
  T4  — Mode A vs Mode C visual grounding
  T5  — Linguistic robustness (paraphrase + negation) — model level
  T6  — Linguistic robustness — per question
  T7  — SCS counterfactual analysis by weather
  T8  — Behavioral classification (affirmation/negation bias per question)
  T9  — Model-level behavioral summary
  T10 — Moondream verbose failure mode classification
  T11 — PaliGemma refusal pattern analysis

Usage: python3 extract_aw_tables_final.py > aw_tables.tsv
"""

import csv, json, os
from collections import defaultdict, Counter

RESULTS  = '/home2/muskan.singh/results'
MODELS   = ['moondream', 'paligemma', 'smolvlm', 'llava_ov', 'internvl3']
MODEL_LABELS = {
    'moondream': 'Moondream 2B',
    'paligemma': 'PaliGemma 3B',
    'smolvlm'  : 'SmolVLM 2.2B',
    'llava_ov' : 'LLaVA-OV 8B',
    'internvl3': 'InternVL3 8B',
}
WEATHERS  = ['rainy', 'foggy', 'snowy', 'night', 'clear']
Q_TYPES   = ['perception', 'action', 'hazard', 'scene', 'adversarial']

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def pct(n, d):
    return round(n / d * 100, 1) if d else 0.0

def tsv(*cols):
    return '\t'.join(str(c) for c in cols)

def sep(title):
    print('\n\n' + '='*80)
    print(title)
    print('='*80)

# ── Load all data ─────────────────────────────────────────────
main_data  = {}   # model → main_eval original rows
modec_data = {}   # model → mode C main_eval original rows
all_data   = {}   # model → all rows

for m in MODELS:
    pa = os.path.join(RESULTS, f'adverse_weather_{m}_v2.csv')
    pc = os.path.join(RESULTS, f'adverse_weather_{m}_noimage_v2.csv')
    if not os.path.exists(pa):
        continue
    rows = load_csv(pa)
    all_data[m]  = rows
    main_data[m] = [r for r in rows
                    if r['use_for'] == 'main_eval'
                    and r['is_augmented'] == 'False']
    if os.path.exists(pc):
        mc = load_csv(pc)
        modec_data[m] = [r for r in mc
                         if r['use_for'] == 'main_eval'
                         and r['is_augmented'] == 'False']

# Load analysis JSONs
with open(os.path.join(RESULTS, 'aw_linguistic_analysis.json')) as f:
    ling = json.load(f)
with open(os.path.join(RESULTS, 'aw_cf_analysis.json')) as f:
    cf = json.load(f)

# Build q_info from data
q_info = {}
for m in MODELS:
    if m not in main_data: continue
    for r in main_data[m]:
        if r['q_id'] not in q_info:
            q_info[r['q_id']] = {
                'question': r['question'],
                'q_type'  : r['q_type'],
                'gt'      : r['ground_truth'],
            }
q_ids = sorted(q_info.keys())

# ─────────────────────────────────────────────────────────────
# T1: Model-level summary
# ─────────────────────────────────────────────────────────────
sep('T1: Model-Level Summary (main_eval, original images only)')
print(tsv('Model', 'Total Rows', 'Overall Acc%',
          'GT=yes Acc%', 'GT=no Acc%', 'Affirmation Gap (yes-no)',
          'Unclear%', 'Avg Response Time(s)', 'Bias Direction'))

for m in MODELS:
    if m not in main_data: continue
    rows  = main_data[m]
    total = len(rows)
    correct  = sum(1 for r in rows if r['correct'].strip().lower() == 'true')
    yes_rows = [r for r in rows if r['ground_truth'].strip().lower() == 'yes']
    no_rows  = [r for r in rows if r['ground_truth'].strip().lower() == 'no']
    yes_c    = sum(1 for r in yes_rows if r['correct'].strip().lower() == 'true')
    no_c     = sum(1 for r in no_rows  if r['correct'].strip().lower() == 'true')
    unclear  = sum(1 for r in rows if r['extracted_ans'].strip().lower() == 'unclear')
    avg_t    = sum(float(r['response_time']) for r in rows if r['response_time']) / total

    overall  = pct(correct, total)
    yes_acc  = pct(yes_c, len(yes_rows))
    no_acc   = pct(no_c,  len(no_rows))
    gap      = round(yes_acc - no_acc, 1)
    unc_pct  = pct(unclear, total)

    if gap > 15:   bias = 'Affirmation bias'
    elif gap < -15: bias = 'Negation bias'
    else:           bias = 'Balanced'

    print(tsv(MODEL_LABELS[m], total, overall, yes_acc, no_acc,
              gap, unc_pct, round(avg_t, 2), bias))

# ─────────────────────────────────────────────────────────────
# T2: Per-question accuracy
# ─────────────────────────────────────────────────────────────
sep('T2: Per-Question Accuracy % (main_eval, original images)')
header = ['Q_ID', 'Q_Type', 'GT', 'Question'] + \
         [MODEL_LABELS[m] for m in MODELS if m in main_data] + ['Mean', 'Min', 'Max']
print(tsv(*header))

for qid in q_ids:
    info = q_info[qid]
    vals = []
    for m in MODELS:
        if m not in main_data: continue
        rows_q = [r for r in main_data[m] if r['q_id'] == qid]
        c = sum(1 for r in rows_q if r['correct'].strip().lower() == 'true')
        vals.append(pct(c, len(rows_q)))
    mean = round(sum(vals)/len(vals), 1) if vals else 0
    mn   = min(vals) if vals else 0
    mx   = max(vals) if vals else 0
    print(tsv(qid, info['q_type'], info['gt'], info['question']) +
          '\t' + '\t'.join(str(v) for v in vals) +
          '\t' + tsv(mean, mn, mx))

# ─────────────────────────────────────────────────────────────
# T3: Per-weather accuracy
# ─────────────────────────────────────────────────────────────
sep('T3: Per-Weather Accuracy % (main_eval, original images)')
print(tsv('Model', *[w.capitalize() for w in WEATHERS],
          'Best Weather', 'Worst Weather', 'Range'))

for m in MODELS:
    if m not in main_data: continue
    w_accs = {}
    for w in WEATHERS:
        wr = [r for r in main_data[m] if r['weather'] == w]
        wc = sum(1 for r in wr if r['correct'].strip().lower() == 'true')
        w_accs[w] = pct(wc, len(wr))
    best  = max(w_accs, key=w_accs.get)
    worst = min(w_accs, key=w_accs.get)
    rng   = round(w_accs[best] - w_accs[worst], 1)
    print(tsv(MODEL_LABELS[m],
              *[w_accs[w] for w in WEATHERS],
              best.capitalize(), worst.capitalize(), rng))

# Per-weather per-qtype breakdown
print()
print(tsv('Model', 'Q_Type', *[w.capitalize() for w in WEATHERS]))
for m in MODELS:
    if m not in main_data: continue
    for qt in Q_TYPES:
        row = [MODEL_LABELS[m], qt]
        for w in WEATHERS:
            wr = [r for r in main_data[m]
                  if r['weather'] == w and r['q_type'] == qt]
            if wr:
                wc = sum(1 for r in wr if r['correct'].strip().lower() == 'true')
                row.append(pct(wc, len(wr)))
            else:
                row.append('—')
        print(tsv(*row))

# ─────────────────────────────────────────────────────────────
# T4: Mode A vs Mode C
# ─────────────────────────────────────────────────────────────
sep('T4: Mode A vs Mode C — Visual Grounding Analysis')
print(tsv('Model',
          'Mode A Acc%', 'Mode C Acc%', 'VGS (A-C)',
          'Mode A GT=yes%', 'Mode A GT=no%',
          'Mode C GT=yes%', 'Mode C GT=no%',
          'Mode C Default Answer', 'Mode C Default Pct%',
          'Interpretation'))

for m in MODELS:
    if m not in main_data or m not in modec_data: continue
    a = main_data[m]
    c = modec_data[m]

    def acc(rows):
        return pct(sum(1 for r in rows if r['correct'].strip().lower()=='true'), len(rows))
    def acc_gt(rows, gt):
        sub = [r for r in rows if r['ground_truth'].strip().lower()==gt]
        return pct(sum(1 for r in sub if r['correct'].strip().lower()=='true'), len(sub))

    a_acc   = acc(a)
    c_acc   = acc(c)
    vgs     = round(a_acc - c_acc, 1)
    a_yes   = acc_gt(a, 'yes')
    a_no    = acc_gt(a, 'no')
    c_yes   = acc_gt(c, 'yes')
    c_no    = acc_gt(c, 'no')

    c_ans   = Counter(r['extracted_ans'].strip().lower() for r in c)
    default = c_ans.most_common(1)[0]
    def_pct = pct(default[1], len(c))

    # Interpretation
    if default[0] == 'no' and def_pct > 80:
        interp = 'Confounded — defaults to NO (VGS unreliable)'
    elif vgs > 15:
        interp = 'Visually grounded (VGS > 15)'
    elif vgs > 5:
        interp = 'Moderate visual grounding'
    else:
        interp = 'Minimal visual grounding'

    print(tsv(MODEL_LABELS[m],
              a_acc, c_acc, vgs,
              a_yes, a_no, c_yes, c_no,
              default[0], def_pct, interp))

# ─────────────────────────────────────────────────────────────
# T5: Linguistic robustness — model level
# ─────────────────────────────────────────────────────────────
sep('T5: Linguistic Robustness — Model Level')
print(tsv('Model', 'Paraphrase Consistency%', 'Negation Accuracy%',
          'Negation vs Chance', 'Interpretation'))

for m in MODELS:
    if m not in ling: continue
    para = [v['pct'] for v in ling[m]['paraphrase_consistency'].values()
            if v['pct'] is not None]
    neg  = [v['pct'] for v in ling[m]['negation_accuracy'].values()
            if v['pct'] is not None]
    pa   = round(sum(para)/len(para), 1) if para else 0
    na   = round(sum(neg)/len(neg),   1) if neg  else 0
    vs_c = round(na - 50.0, 1)

    if na < 45:   interp = 'Severe negation blindness (below chance)'
    elif na < 55: interp = 'Near-chance negation (systematic failure)'
    else:         interp = 'Above chance on negation'

    print(tsv(MODEL_LABELS[m], pa, na,
              str(vs_c) + '%', interp))

# ─────────────────────────────────────────────────────────────
# T6: Linguistic robustness — per question
# ─────────────────────────────────────────────────────────────
sep('T6: Linguistic Robustness — Per Question')
m_in_ling = [m for m in MODELS if m in ling]
header = ['Q_ID', 'Q_Type', 'GT'] + \
         [MODEL_LABELS[m] + ' Para%' for m in m_in_ling] + \
         ['Para Mean'] + \
         [MODEL_LABELS[m] + ' Neg%'  for m in m_in_ling] + \
         ['Neg Mean']
print(tsv(*header))

for qid in q_ids:
    info = q_info[qid]
    para_vals, neg_vals = [], []
    for m in m_in_ling:
        pv = ling[m]['paraphrase_consistency'].get(qid, {}).get('pct')
        nv = ling[m]['negation_accuracy'].get(qid, {}).get('pct')
        para_vals.append(pv if pv is not None else '—')
        neg_vals.append(nv  if nv is not None else '—')

    p_nums = [v for v in para_vals if isinstance(v, float)]
    n_nums = [v for v in neg_vals  if isinstance(v, float)]
    p_mean = round(sum(p_nums)/len(p_nums), 1) if p_nums else '—'
    n_mean = round(sum(n_nums)/len(n_nums), 1) if n_nums else '—'

    print(tsv(qid, info['q_type'], info['gt'],
              *para_vals, p_mean, *neg_vals, n_mean))

# ─────────────────────────────────────────────────────────────
# T7: SCS counterfactual
# ─────────────────────────────────────────────────────────────
sep('T7: Scene-Change Sensitivity (SCS) — Counterfactual Analysis')
print('Note: Snow SCS uses SD augmentation (JarvisIR snow too subtle per Shankar feedback)')
print(tsv('Model', 'Fog SCS%', 'Rain SCS%', 'Snow SCS% (SD)',
          'Overall SCS%', 'Total Pairs',
          'Both Correct', 'Both Wrong',
          'Correct→Wrong (degradation)',
          'Wrong→Correct (improvement)',
          'Net Effect'))

for m in MODELS:
    if m not in cf: continue
    d    = cf[m]
    fog  = d.get('scs_by_weather', {}).get('foggy',  '—')
    rain = d.get('scs_by_weather', {}).get('rainy',  '—')
    snow = d.get('scs_by_weather', {}).get('snowy',  '—')
    overall = d.get('scs_overall', '—')
    pairs   = d.get('n_pairs', '—')
    dirs    = d.get('directions', {})
    bc  = dirs.get('both_correct',     '—')
    bw  = dirs.get('both_wrong',       '—')
    c2w = dirs.get('correct_to_wrong', '—')
    w2c = dirs.get('wrong_to_correct', '—')
    net = (w2c - c2w) if isinstance(w2c, int) and isinstance(c2w, int) else '—'
    net_str = ('+' + str(net) if isinstance(net, int) and net > 0
               else str(net) if isinstance(net, int) else '—')
    print(tsv(MODEL_LABELS[m], fog, rain, snow, overall,
              pairs, bc, bw, c2w, w2c, net_str))

# ─────────────────────────────────────────────────────────────
# T8: Behavioral classification per question
# ─────────────────────────────────────────────────────────────
sep('T8: Behavioral Classification Per Question — All Models')
print('AFFIRMATION_BIAS: GT=no acc <30% AND GT=yes acc >60%')
print('NEGATION_BIAS:    GT=yes acc <30% AND GT=no acc >60%')
print('MIXED_FAILURE:    both GT=yes and GT=no acc <50%')
print('BALANCED:         neither strong bias')
print()

def behavioral_per_q(rows):
    stats = defaultdict(lambda: {'y_c':0,'y_t':0,'n_c':0,'n_t':0})
    for r in rows:
        if r['extracted_ans'].strip().lower() not in ('yes','no'): continue
        qid = r['q_id']
        gt  = r['ground_truth'].strip().lower()
        ok  = r['correct'].strip().lower() == 'true'
        if gt == 'yes':
            stats[qid]['y_t'] += 1
            if ok: stats[qid]['y_c'] += 1
        else:
            stats[qid]['n_t'] += 1
            if ok: stats[qid]['n_c'] += 1
    result = {}
    for qid, s in stats.items():
        ya = pct(s['y_c'], s['y_t'])
        na = pct(s['n_c'], s['n_t'])
        if na < 30 and ya > 60:    bias = 'AFFIRMATION_BIAS'
        elif ya < 30 and na > 60:  bias = 'NEGATION_BIAS'
        elif ya < 50 and na < 50:  bias = 'MIXED_FAILURE'
        else:                       bias = 'BALANCED'
        result[qid] = {'ya': ya, 'na': na, 'bias': bias}
    return result

m_in = [m for m in MODELS if m in main_data]
header = (['Q_ID', 'Q_Type', 'GT'] +
          [MODEL_LABELS[m] + ' Bias' for m in m_in] +
          [MODEL_LABELS[m] + ' GT=yes%' for m in m_in] +
          [MODEL_LABELS[m] + ' GT=no%'  for m in m_in])
print(tsv(*header))

beh_all = {m: behavioral_per_q(main_data[m]) for m in m_in}

for qid in q_ids:
    info = q_info[qid]
    biases  = [beh_all[m].get(qid, {}).get('bias', '—') for m in m_in]
    ya_vals = [str(beh_all[m].get(qid, {}).get('ya',  '—')) for m in m_in]
    na_vals = [str(beh_all[m].get(qid, {}).get('na',  '—')) for m in m_in]
    print(tsv(qid, info['q_type'], info['gt'],
              *biases, *ya_vals, *na_vals))

# ─────────────────────────────────────────────────────────────
# T9: Model-level behavioral summary
# ─────────────────────────────────────────────────────────────
sep('T9: Model-Level Behavioral Summary')
print(tsv('Model', 'Total Questions',
          'Affirmation Bias Qs', 'Negation Bias Qs',
          'Mixed Failure Qs', 'Balanced Qs',
          'Dominant Failure', 'Safety-Critical Bias Qs (hazard+action)'))

for m in m_in:
    b = beh_all[m]
    counts = Counter(v['bias'] for v in b.values())
    total_q = len(b)
    # Safety-critical = hazard + action questions with affirmation bias
    sc_bias = sum(1 for qid, v in b.items()
                  if v['bias'] == 'AFFIRMATION_BIAS'
                  and q_info.get(qid, {}).get('q_type','') in ('hazard','action'))
    dominant = counts.most_common(1)[0][0] if counts else '—'
    print(tsv(MODEL_LABELS[m], total_q,
              counts.get('AFFIRMATION_BIAS', 0),
              counts.get('NEGATION_BIAS',    0),
              counts.get('MIXED_FAILURE',    0),
              counts.get('BALANCED',         0),
              dominant, sc_bias))

# ─────────────────────────────────────────────────────────────
# T10: Moondream verbose failure mode classification
# ─────────────────────────────────────────────────────────────
sep('T10: Moondream 2B — Verbose Response Failure Mode Classification')
print('Note: Only Moondream received unconstrained prompts — other models given yes/no only instruction')
print()

VISUAL_EVIDENCE = [
    'i can see','visible','i see','image shows','in the image',
    'appears to be','the scene shows','from the image',
    'the road appears','i can observe','the picture','depicted'
]
WEATHER_KW = {
    'rain' : ['rain','raindrops','raining','rainfall','wet windshield','rainy'],
    'fog'  : ['fog','foggy','mist','misty','haze'],
    'snow' : ['snow','snowing','snowflakes','snow-covered','blizzard','snowy'],
    'night': ['night','dark','darkness','nighttime','low light'],
    'clear': ['clear','sunny','bright','daylight'],
}
SAFETY_PRIOR = [
    'safety precautions should always','it is always important',
    'drivers should always','as a general rule',
    'it is essential to always','in any driving situation',
]
HEDGES = [
    'difficult to see','hard to tell','not possible to determine',
    'limited visibility','cannot be seen','not clearly visible',
    'unclear from the image','difficult to determine'
]

def classify_moon(row):
    resp    = row['full_response'].lower().strip()
    gt      = row['ground_truth'].strip().lower()
    ans     = row['extracted_ans'].strip().lower()
    correct = row['correct'].strip().lower() == 'true'
    weather = row['weather'].strip().lower()
    vis     = [kw for kw in VISUAL_EVIDENCE if kw in resp]

    if ans == 'unclear':
        neg_s = sum(1 for p in ['not possible','cannot','unable','not safe',
                                 'should not','is not','are not','no,'] if p in resp)
        pos_s = sum(1 for p in ['yes,','it is','there is','the road is',
                                 'the driver should','it appears'] if p in resp)
        if neg_s > pos_s: return 'UNCLEAR_RECOVERABLE_NO'
        if pos_s > neg_s: return 'UNCLEAR_RECOVERABLE_YES'
        return 'UNCLEAR_UNRECOVERABLE'

    if correct:
        wrong_wx = any(
            w not in weather and any(kw in resp for kw in kws)
            for w, kws in WEATHER_KW.items()
        )
        if wrong_wx: return 'LUCKY_CORRECT'
        if any(p in resp for p in SAFETY_PRIOR): return 'PRIOR_CORRECT'
        if vis: return 'VISUAL_GROUNDED'
        return 'PRIOR_CORRECT'

    if any(p in resp for p in SAFETY_PRIOR): return 'SAFETY_PRIOR'
    for w, kws in WEATHER_KW.items():
        if w not in weather and any(kw in resp for kw in kws):
            return 'WEATHER_CONFUSION'
    hallu = ['pedestrian','person crossing','red light','green light',
             'traffic light','stop sign','vehicle ahead','car in front']
    if any(h in resp for h in hallu) and gt == 'no' and ans == 'yes':
        return 'HALLUCINATION'
    if any(h in resp for h in HEDGES): return 'CORRECT_REASONING_WRONG_ANS'
    if vis: return 'HALLUCINATION'
    return 'LANGUAGE_PRIOR'

MODES = ['VISUAL_GROUNDED','PRIOR_CORRECT','LUCKY_CORRECT',
         'HALLUCINATION','WEATHER_CONFUSION','SAFETY_PRIOR',
         'CORRECT_REASONING_WRONG_ANS','LANGUAGE_PRIOR',
         'UNCLEAR_RECOVERABLE_YES','UNCLEAR_RECOVERABLE_NO','UNCLEAR_UNRECOVERABLE']

DESCRIPTIONS = {
    'VISUAL_GROUNDED'            : 'Correct + cites specific visual evidence',
    'PRIOR_CORRECT'              : 'Correct but reasoning generic/text-based',
    'LUCKY_CORRECT'              : 'Correct but describes wrong weather conditions',
    'HALLUCINATION'              : 'Wrong + fabricates visual content not in image',
    'WEATHER_CONFUSION'          : 'Wrong + describes incorrect weather type',
    'SAFETY_PRIOR'               : 'Wrong + generic safety language ignoring scene',
    'CORRECT_REASONING_WRONG_ANS': 'Reasoning directionally correct but answer wrong',
    'LANGUAGE_PRIOR'             : 'Wrong + no visual grounding, text-based only',
    'UNCLEAR_RECOVERABLE_YES'    : 'Unclear extraction — response implies YES',
    'UNCLEAR_RECOVERABLE_NO'     : 'Unclear extraction — response implies NO',
    'UNCLEAR_UNRECOVERABLE'      : 'Cannot determine intended answer from response',
}

if 'moondream' in main_data:
    moon_rows = main_data['moondream']
    classified = [(r, classify_moon(r)) for r in moon_rows]
    counts = Counter(m for _, m in classified)
    total  = len(moon_rows)

    print(tsv('Mode', 'Count', 'Pct%', 'Category', 'Description'))
    for mode in MODES:
        n    = counts.get(mode, 0)
        cat  = ('Correct' if mode in ('VISUAL_GROUNDED','PRIOR_CORRECT','LUCKY_CORRECT')
                else 'Wrong' if 'UNCLEAR' not in mode else 'Unclear')
        print(tsv(mode, n, str(pct(n, total))+'%', cat, DESCRIPTIONS.get(mode,'')))

    # Per-question dominant failure mode
    print()
    print(tsv('Q_ID', 'Q_Type', 'Total Wrong', 'Dominant Failure Mode',
              'Pct%', 'GT=yes Acc%', 'GT=no Acc%', 'Interpretation'))
    q_wrong = defaultdict(list)
    for r, mode in classified:
        if r['correct'].strip().lower() == 'false':
            q_wrong[r['q_id']].append(mode)

    beh_moon = behavioral_per_q(moon_rows)
    for qid in sorted(q_wrong.keys()):
        wrongs = q_wrong[qid]
        if not wrongs: continue
        dom  = Counter(wrongs).most_common(1)[0]
        b    = beh_moon.get(qid, {})
        ya   = b.get('ya', '—')
        na   = b.get('na', '—')
        # Interpretation
        if dom[0] == 'HALLUCINATION':
            interp = 'Model fabricates visual content to justify wrong answer'
        elif dom[0] == 'WEATHER_CONFUSION':
            interp = 'Model misidentifies weather type — confuses scene context'
        elif dom[0] == 'LANGUAGE_PRIOR':
            interp = 'Answer driven by question framing not visual content'
        elif dom[0] == 'CORRECT_REASONING_WRONG_ANS':
            interp = 'Model understands scene but extracts wrong yes/no'
        elif dom[0] == 'LUCKY_CORRECT':
            interp = 'Correct answer despite wrong weather description'
        else:
            interp = ''
        print(tsv(qid, q_info.get(qid,{}).get('q_type',''),
                  len(wrongs), dom[0],
                  str(round(dom[1]/len(wrongs)*100))+'%',
                  ya, na, interp))

# ─────────────────────────────────────────────────────────────
# T11: PaliGemma refusal pattern
# ─────────────────────────────────────────────────────────────
sep('T11: PaliGemma 3B — Refusal & Unusual Response Analysis')
print('Note: PaliGemma is the only model that sometimes refuses or says unanswerable')
print()

if 'paligemma' in main_data:
    pg_rows = main_data['paligemma']
    refusals = [r for r in pg_rows
                if 'not trained' in r['full_response'].lower()
                or 'unanswerable' in r['full_response'].lower()
                or 'sorry' in r['full_response'].lower()]
    non_yn   = [r for r in pg_rows
                if r['extracted_ans'].strip().lower() not in ('yes','no','unclear')
                and len(r['full_response'].strip()) > 3]

    print(tsv('Category', 'Count', 'Pct%', 'Note'))
    print(tsv('Total responses',  len(pg_rows), '100%', ''))
    print(tsv('Bare yes/no',
              sum(1 for r in pg_rows if r['extracted_ans'] in ('yes','no')),
              pct(sum(1 for r in pg_rows if r['extracted_ans'] in ('yes','no')), len(pg_rows)),
              'Standard responses'))
    print(tsv('Refusals',  len(refusals),
              pct(len(refusals), len(pg_rows)),
              '"Sorry, as a base VLM I am not trained to answer this question"'))
    print(tsv('Unanswerable', sum(1 for r in pg_rows if 'unanswerable' in r['full_response'].lower()),
              pct(sum(1 for r in pg_rows if 'unanswerable' in r['full_response'].lower()), len(pg_rows)),
              'Model acknowledges it cannot determine answer from image'))

    if refusals:
        print()
        print('Refusal distribution by Q_Type:')
        ref_by_type = Counter(r['q_type'] for r in refusals)
        print(tsv('Q_Type', 'Refusal Count'))
        for qt, cnt in ref_by_type.most_common():
            print(tsv(qt, cnt))

print('\n\nDone. Copy each TABLE section into a separate Google Sheets tab.')
print('Tables: T1=Summary, T2=PerQuestion, T3=PerWeather, T4=ModeAC,')
print('        T5=LingModel, T6=LingQuestion, T7=SCS, T8=BehavioralQ,')
print('        T9=BehavioralModel, T10=MoondreamVerbose, T11=PaliGemmaRefusal')
