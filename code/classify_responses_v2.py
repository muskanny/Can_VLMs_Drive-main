"""
classify_responses_v2.py
------------------------
Two-track classification:
  Track 1 (Moondream only): verbose response text classification
  Track 2 (all models): behavioral pattern classification from answer patterns
"""
import csv, json, os, re
from collections import defaultdict, Counter

RESULTS = '/home2/muskan.singh/results'
MODELS  = ['moondream', 'paligemma', 'smolvlm', 'llava_ov', 'internvl3']
MODEL_LABELS = {
    'moondream': 'Moondream 2B', 'paligemma': 'PaliGemma 3B',
    'smolvlm'  : 'SmolVLM 2.2B', 'llava_ov' : 'LLaVA-OV 8B',
    'internvl3': 'InternVL3 8B',
}

def pct(n, d):
    return round(n/d*100, 1) if d else 0.0

# ── Track 1: Moondream verbose classifier ─────────────────────
VISUAL_EVIDENCE = [
    'i can see', 'visible', 'i see', 'image shows', 'in the image',
    'appears to be', 'the scene shows', 'looking at', 'the photo',
    'depicted', 'shown in', 'from the image', 'the road appears',
    'i can observe', 'the picture'
]
WEATHER_KEYWORDS = {
    'rain' : ['rain', 'raindrops', 'raining', 'rainfall', 'wet windshield', 'rainy'],
    'fog'  : ['fog', 'foggy', 'mist', 'misty', 'haze', 'foggy'],
    'snow' : ['snow', 'snowing', 'snowflakes', 'snow-covered', 'blizzard', 'snowy'],
    'night': ['night', 'dark', 'darkness', 'nighttime', 'low light'],
    'clear': ['clear', 'sunny', 'bright', 'daylight', 'clear sky'],
}
SAFETY_PRIOR = [
    'safety precautions should always', 'it is always important',
    'drivers should always', 'as a general rule',
    'it is essential to always', 'in any driving situation',
]

def classify_moondream(row):
    resp    = row['full_response'].lower().strip()
    gt      = row['ground_truth'].strip().lower()
    ans     = row['extracted_ans'].strip().lower()
    correct = row['correct'].strip().lower() == 'true'
    weather = row['weather'].strip().lower()

    visual_found = [kw for kw in VISUAL_EVIDENCE if kw in resp]

    # Unclear
    if ans == 'unclear':
        neg_phrases = ['not possible','cannot','unable to','not safe',
                       'should not','is not','are not','no,','this is not']
        pos_phrases = ['yes,','it is','there is','the road is',
                       'the driver should','it appears']
        neg_score = sum(1 for p in neg_phrases if p in resp)
        pos_score = sum(1 for p in pos_phrases if p in resp)
        if neg_score > pos_score: return 'UNCLEAR_RECOVERABLE_NO'
        if pos_score > neg_score: return 'UNCLEAR_RECOVERABLE_YES'
        return 'UNCLEAR_UNRECOVERABLE'

    # Correct
    if correct:
        # Check if model describes wrong weather (lucky correct)
        wrong_wx = False
        for w, kws in WEATHER_KEYWORDS.items():
            if w not in weather and any(kw in resp for kw in kws):
                wrong_wx = True; break
        if wrong_wx: return 'LUCKY_CORRECT'
        if any(p in resp for p in SAFETY_PRIOR): return 'PRIOR_CORRECT'
        if len(visual_found) >= 1: return 'VISUAL_GROUNDED'
        return 'PRIOR_CORRECT'

    # Wrong
    if any(p in resp for p in SAFETY_PRIOR): return 'SAFETY_PRIOR'
    # Weather confusion: model describes a different weather
    for w, kws in WEATHER_KEYWORDS.items():
        if w not in weather and any(kw in resp for kw in kws):
            return 'WEATHER_CONFUSION'
    # Hallucination: model describes specific objects to justify wrong answer
    hallu_words = ['pedestrian','person crossing','red light','green light',
                   'traffic light','stop sign','vehicle ahead','car in front']
    if any(hw in resp for hw in hallu_words) and gt == 'no' and ans == 'yes':
        return 'HALLUCINATION'
    # Correct reasoning, wrong answer
    hedges = ['difficult to see','hard to tell','not possible to determine',
              'limited visibility','cannot be seen','not clearly visible',
              'unclear from the image','difficult to determine']
    if any(h in resp for h in hedges): return 'CORRECT_REASONING_WRONG_ANS'
    if visual_found: return 'HALLUCINATION'
    return 'LANGUAGE_PRIOR'

# ── Track 2: Behavioral classification (all models) ──────────
def classify_behavioral(rows):
    """
    For each question type, classify the error pattern:
    AFFIRMATION_BIAS: GT=no but model says yes consistently
    NEGATION_BIAS: GT=yes but model says no consistently
    MIXED_FAILURE: both directions fail
    VISUAL_FAILURE: fails on specific weather/visual conditions
    """
    stats = defaultdict(lambda: {
        'gt_yes_correct': 0, 'gt_yes_wrong': 0,
        'gt_no_correct' : 0, 'gt_no_wrong' : 0,
        'total': 0
    })
    weather_stats = defaultdict(lambda: defaultdict(lambda: [0,0]))

    for r in rows:
        qid     = r['q_id']
        gt      = r['ground_truth'].strip().lower()
        ans     = r['extracted_ans'].strip().lower()
        correct = r['correct'].strip().lower() == 'true'
        weather = r['weather'].strip().lower()

        if ans not in ('yes','no'): continue
        stats[qid]['total'] += 1
        if gt == 'yes':
            if correct: stats[qid]['gt_yes_correct'] += 1
            else:       stats[qid]['gt_yes_wrong']   += 1
        else:
            if correct: stats[qid]['gt_no_correct']  += 1
            else:       stats[qid]['gt_no_wrong']    += 1

        weather_stats[qid][weather][0 if correct else 1] += 1

    result = {}
    for qid, s in stats.items():
        gt_yes_acc = pct(s['gt_yes_correct'], s['gt_yes_correct']+s['gt_yes_wrong'])
        gt_no_acc  = pct(s['gt_no_correct'],  s['gt_no_correct'] +s['gt_no_wrong'])

        if gt_no_acc < 30 and gt_yes_acc > 60:
            bias = 'AFFIRMATION_BIAS'
        elif gt_yes_acc < 30 and gt_no_acc > 60:
            bias = 'NEGATION_BIAS'
        elif gt_yes_acc < 50 and gt_no_acc < 50:
            bias = 'MIXED_FAILURE'
        else:
            bias = 'BALANCED'

        # Find worst weather
        worst_w = min(weather_stats[qid].items(),
                      key=lambda x: pct(x[1][0], x[1][0]+x[1][1])
                      if (x[1][0]+x[1][1]) > 0 else 100,
                      default=(None, None))

        result[qid] = {
            'gt_yes_acc': gt_yes_acc,
            'gt_no_acc' : gt_no_acc,
            'bias'      : bias,
            'worst_weather': worst_w[0] if worst_w[0] else '—',
        }
    return result

# ── Load and classify all models ─────────────────────────────
all_data = {}
for m in MODELS:
    path = os.path.join(RESULTS, f'adverse_weather_{m}_v2.csv')
    if not os.path.exists(path): continue
    rows = list(csv.DictReader(open(path)))
    main = [r for r in rows if r['use_for'] == 'main_eval'
            and r['is_augmented'] == 'False']
    all_data[m] = main

# ── TABLE 7: Moondream verbose classification ─────────────────
print('='*80)
print('TABLE 7: Moondream 2B — Verbose Response Classification')
print('(Only Moondream provides verbose reasoning — other models give bare yes/no)')
print('='*80)

MOON_MODES = ['VISUAL_GROUNDED','PRIOR_CORRECT','LUCKY_CORRECT',
              'HALLUCINATION','WEATHER_CONFUSION','SAFETY_PRIOR',
              'CORRECT_REASONING_WRONG_ANS','LANGUAGE_PRIOR',
              'UNCLEAR_RECOVERABLE_YES','UNCLEAR_RECOVERABLE_NO','UNCLEAR_UNRECOVERABLE']

moon_rows  = all_data.get('moondream', [])
moon_classified = [(r, classify_moondream(r)) for r in moon_rows]
mode_counts = Counter(m for _, m in moon_classified)
total = len(moon_rows)

print('Mode\tCount\tPct%\tDescription')
descriptions = {
    'VISUAL_GROUNDED'           : 'Correct + cites visual evidence from image',
    'PRIOR_CORRECT'             : 'Correct but reasoning is generic/language-based',
    'LUCKY_CORRECT'             : 'Correct but describes wrong weather conditions',
    'HALLUCINATION'             : 'Wrong + fabricates visual content not in image',
    'WEATHER_CONFUSION'         : 'Wrong + describes incorrect weather type',
    'SAFETY_PRIOR'              : 'Wrong + generic safety language regardless of scene',
    'CORRECT_REASONING_WRONG_ANS': 'Reasoning directionally correct but final answer wrong',
    'LANGUAGE_PRIOR'            : 'Wrong + no visual grounding, text-based response',
    'UNCLEAR_RECOVERABLE_YES'   : 'Unclear extraction but response implies YES',
    'UNCLEAR_RECOVERABLE_NO'    : 'Unclear extraction but response implies NO',
    'UNCLEAR_UNRECOVERABLE'     : 'Cannot determine intended answer',
}
for mode in MOON_MODES:
    n = mode_counts.get(mode, 0)
    print('\t'.join([mode, str(n), str(pct(n,total))+'%', descriptions.get(mode,'')]))

# ── TABLE 7b: Moondream per-question failure mode ─────────────
print()
print('TABLE 7b: Moondream — Dominant Failure Mode Per Question (wrong answers only)')
print('Q_ID\tQ_Type\tTotal Wrong\tDominant Mode\tPct%\tGT=yes Acc%\tGT=no Acc%')

q_wrong = defaultdict(list)
for r, mode in moon_classified:
    if r['correct'].strip().lower() == 'false':
        q_wrong[r['q_id']].append(mode)

q_info = {}
for r in moon_rows:
    if r['q_id'] not in q_info:
        q_info[r['q_id']] = r['q_type']

behav = classify_behavioral(moon_rows)
for qid in sorted(q_wrong.keys()):
    wrongs = q_wrong[qid]
    dom = Counter(wrongs).most_common(1)[0]
    b = behav.get(qid, {})
    print('\t'.join([qid, q_info.get(qid,''), str(len(wrongs)),
                     dom[0], str(round(dom[1]/len(wrongs)*100))+'%',
                     str(b.get('gt_yes_acc','—')), str(b.get('gt_no_acc','—'))]))

# ── TABLE 8: Behavioral analysis — all models ─────────────────
print()
print('='*80)
print('TABLE 8: Behavioral Classification — All Models')
print('AFFIRMATION_BIAS: GT=no acc <30% while GT=yes acc >60%')
print('NEGATION_BIAS: GT=yes acc <30% while GT=no acc >60%')
print('='*80)

# Get all q_ids
all_qids = sorted(set(r['q_id'] for m in all_data for r in all_data[m]))
q_types  = {r['q_id']: r['q_type'] for m in all_data for r in all_data[m]}

header = ['Q_ID','Q_Type'] + \
         [MODEL_LABELS[m] + ' Bias' for m in MODELS if m in all_data] + \
         [MODEL_LABELS[m] + ' GTyes%' for m in MODELS if m in all_data] + \
         [MODEL_LABELS[m] + ' GTno%'  for m in MODELS if m in all_data]
print('\t'.join(header))

for qid in all_qids:
    row = [qid, q_types.get(qid,'')]
    biases, gt_yes_accs, gt_no_accs = [], [], []
    for m in MODELS:
        if m not in all_data: continue
        b = classify_behavioral(all_data[m])
        info = b.get(qid, {'bias':'—','gt_yes_acc':'—','gt_no_acc':'—'})
        biases.append(info['bias'])
        gt_yes_accs.append(str(info['gt_yes_acc']))
        gt_no_accs.append(str(info['gt_no_acc']))
    print('\t'.join(row + biases + gt_yes_accs + gt_no_accs))

# ── TABLE 9: Model-level behavioral summary ───────────────────
print()
print('='*80)
print('TABLE 9: Model-Level Behavioral Summary')
print('='*80)
print('\t'.join(['Model','Total Qs','Affirmation Bias Qs','Negation Bias Qs',
                 'Mixed Failure Qs','Balanced Qs','Most Common Failure']))

for m in MODELS:
    if m not in all_data: continue
    b = classify_behavioral(all_data[m])
    counts = Counter(v['bias'] for v in b.values())
    total_q = len(b)
    most_common = counts.most_common(1)[0][0] if counts else '—'
    print('\t'.join([MODEL_LABELS[m], str(total_q),
                     str(counts.get('AFFIRMATION_BIAS',0)),
                     str(counts.get('NEGATION_BIAS',0)),
                     str(counts.get('MIXED_FAILURE',0)),
                     str(counts.get('BALANCED',0)),
                     most_common]))

print('\nDone.')
