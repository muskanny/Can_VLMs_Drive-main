"""
classify_responses.py
---------------------
Classifies full_response for each row into failure/success modes.
Does NOT re-run inference — works entirely on existing full_response column.

Failure modes (wrong answers):
  HALLUCINATION     — model describes visual content not present in image
  WEATHER_CONFUSION — model identifies wrong weather type
  SAFETY_PRIOR      — model defaults to safety answer regardless of image
  CORRECT_REASONING — reasoning is directionally correct but final answer wrong
  LANGUAGE_PRIOR    — generic text-based response, no visual grounding
  OTHER_WRONG       — wrong but doesn't fit above categories

Success modes (correct answers):
  VISUAL_GROUNDED   — cites specific visual evidence from image
  PRIOR_CORRECT     — correct answer but reasoning is generic/language-prior
  LUCKY_CORRECT     — reasoning describes wrong conditions but answer happens correct

Unclear classification:
  RECOVERABLE_YES   — answer is yes embedded in description
  RECOVERABLE_NO    — answer is no embedded in description
  TRULY_UNCLEAR     — cannot determine intended answer
"""

import csv, json, os, re
from collections import defaultdict, Counter

RESULTS  = '/home2/muskan.singh/results'
MODELS   = ['moondream', 'paligemma', 'smolvlm', 'llava_ov', 'internvl3']

# ── Keyword lists ─────────────────────────────────────────────

# Visual evidence words — model is citing what it sees
VISUAL_EVIDENCE = [
    'i can see', 'visible', 'i see', 'image shows', 'in the image',
    'appears to be', 'the scene shows', 'looking at', 'the photo',
    'the picture', 'depicted', 'shown in', 'clearly shows',
    'from the image', 'the road appears', 'i can observe'
]

# Hallucination markers — specific objects/conditions that may not be present
HALLUCINATION_OBJECTS = {
    'pedestrian': ['pedestrian', 'person crossing', 'people walking', 'walker'],
    'rain':       ['rain', 'raindrops', 'raining', 'rainfall', 'wet windshield'],
    'snow':       ['snow', 'snowing', 'snowflakes', 'snow-covered', 'blizzard'],
    'fog':        ['fog', 'foggy', 'mist', 'misty', 'haze'],
    'traffic_light': ['traffic light', 'traffic signal', 'red light', 'green light'],
}

# Safety prior markers — generic safety language not grounded in visual content
SAFETY_PRIOR_PHRASES = [
    'safety precautions should always',
    'it is always important',
    'drivers should always',
    'as a general rule',
    'it is essential to always',
    'in any driving situation',
    'regardless of conditions',
    'it is recommended to always',
]

# Language prior markers — answer based on question framing not visual content
LANGUAGE_PRIOR_PHRASES = [
    'based on the question',
    'the question asks',
    'as stated',
    'generally speaking',
    'in most cases',
    'typically',
    'usually drivers',
]

def classify_response(row):
    """
    Returns (failure_mode, confidence, visual_keywords_found)
    """
    resp   = row['full_response'].lower().strip()
    gt     = row['ground_truth'].strip().lower()
    ans    = row['extracted_ans'].strip().lower()
    correct = row['correct'].strip().lower() == 'true'
    weather = row['weather'].strip().lower()
    q_type  = row['q_type'].strip().lower()

    visual_found = [kw for kw in VISUAL_EVIDENCE if kw in resp]

    # ── UNCLEAR classification ────────────────────────────────
    if ans == 'unclear':
        # Try to recover answer
        yes_count = resp.count(' yes') + (1 if resp.startswith('yes') else 0)
        no_count  = resp.count(' no')  + (1 if resp.startswith('no')  else 0)
        # Strong negative indicators
        neg_phrases = ['not possible', 'cannot', 'unable to', 'not safe',
                       'should not', 'is not', 'are not', 'does not',
                       'this is not', 'it is not', 'no,']
        pos_phrases = ['yes,', 'it is', 'there is', 'the road is',
                       'the driver should', 'it appears', 'it seems']
        neg_score = sum(1 for p in neg_phrases if p in resp)
        pos_score = sum(1 for p in pos_phrases if p in resp)

        if neg_score > pos_score or no_count > yes_count:
            mode = 'RECOVERABLE_NO'
        elif pos_score > neg_score or yes_count > no_count:
            mode = 'RECOVERABLE_YES'
        else:
            mode = 'TRULY_UNCLEAR'
        return mode, 'medium', visual_found

    # ── CORRECT classification ────────────────────────────────
    if correct:
        if len(visual_found) >= 2:
            return 'VISUAL_GROUNDED', 'high', visual_found
        # Check if weather description matches actual weather
        weather_match = weather in resp
        if weather_match and visual_found:
            return 'VISUAL_GROUNDED', 'medium', visual_found
        # Check for safety prior (correct by coincidence)
        if any(p in resp for p in SAFETY_PRIOR_PHRASES):
            return 'PRIOR_CORRECT', 'high', visual_found
        # Check if model describes wrong weather but answer still correct
        wrong_weather = False
        for w, kws in HALLUCINATION_OBJECTS.items():
            if w != weather and any(kw in resp for kw in kws):
                wrong_weather = True
                break
        if wrong_weather:
            return 'LUCKY_CORRECT', 'medium', visual_found
        if visual_found:
            return 'VISUAL_GROUNDED', 'medium', visual_found
        return 'PRIOR_CORRECT', 'low', visual_found

    # ── WRONG classification ────────────────────────────

�    # Safety prior — generic safety language
    if any(
        return 'SAFETY_PRIOR', 'high', visual_found

    
 
ms():
        if w != weather and w not in ('pedestrian', 'traffic_light'):
    
                wrong_wea
    if wrong_weather_described:
        return 'WEATHER_CONFUSION', 'high', visu

    # Hallucination — model describes objects/conditions not matching scene
  
    if gt == 'no':  # model said yes but GT is no
        if ans == 'yes':
     
            hallu_found = []
            for obj, kws in HAL
                if any(kw in resp for kw in kws):
                    hallu_found.app
 

            if hallu_found:
                return 'HALLUCINATION', 'medium', vis

    # Correct reasoning
    # Model shows understanding but giv
    reasoning_correct_phrases
        'difficult to see', 
        'limited visibil
        'unclear from the image'
    ]
    if any(p in resp for 
        return 'CORRECT_REASONI

    # Language prior — generic 
 


    # Default
    if visual_found:
        return 'HALLUCINATION', 'low', visua
    return 'LANGUAGE_PRIOR


# ── Run classificatio
print('Running response 
print()

# Summa
MODES = ['VISUAL_GROUNDED
         'HALLUCINAT
         'CORREC
         'RECOVERABLE_

DEL_LABELS = {
    'moondream
    'pa
    'smolvlm'  : 'SmolVLM 2.2B',
    'llava_ov' : 'LLaVA-OV 8
    'in
}

print('='*80)
print('TABLE 7: Response Classification — Failure & Success Modes')
print('Corr
print('Wrong answers:   HALLUCINATION | WEATHER_CONFUSION | SAFETY_PRIOR | CORRECT_REASONING_WRONG_ANS | LANGUAGE_PRIOR')
print('Unclear answers: RECOVERABLE_YES | RECOVERABLE_NO | TRULY_UNCLEAR')
print('='*80)
print('\t'.join(['Model', 'Total'] + MODES))

a

for m in MODELS:
    path = os.path.join(RESULTS, f'adverse_weather_{m}_v2.csv')
    if not os.path.exists(path): continue
    rows = list(csv.DictReader(open(path)))
    main = [r for r in'] == 'main_eval'
            and r['is_augmented'] == 'False']

    mode_counts = Counter()
    classified_rows = []
    for r in main:
        mode, conf,
        mode_counts[mode] += 1
        classified_rows.append({**r, 'response_mode': mode, 'confidence': conf,
                                  'visual_keywords': '|'.join(vis)})

    all_model_stats[m] = {'counts': mode_counts, 'total':

    total = len(main)
    row_vals = [MODEL_LABELS[m], total]
   
        n = mode_counts.get(mode, 0)
        row_vals.append(str(n) + ' (' + str(round(n/total*100,1)) + '%)')
    print('\t'.join

# ── Per-question failure mode breakdown ────

�print('='*80)
print('TABLE 8: Per-Question Dominant Failung answers only)')
print('='*80)

# 
q_ids_all = sorted(set(
    r['q_id'] for m in all_model_st
    for r in all_model_stats[m]['row
))

print('\t
for qid in q_ids_all:
    row_out = [qid]
    q_type = ''
    for m in MODELS:
        if m not in all_model_stats: continue
        wrong = [r for r in all_model_stats[m]['rows']
                 if r['q_id'] == qid
            r['correct'].strip().lower() == 'false'
                 and r['extracted_ans'] != 'unclear']
        q_type = wrong[0]['q_type'] if wrong else ''
        
            mode_count = Counter(r['res
            dominant = mode_count.most_com
            row_out.append(dominant[0] + '(' + str(round(dominant[1]/len(wrong)*100)) + '%)')
        else:
          
    row_out.insert(1, q_type)
    print('\t'.join(row_ou

# ── Correct answer grounding analysis ────
print()
print('='*80)
print('TABLE 9: Correct Answer Quality — Visual G
print('='*80)
print('\t'.join(['Model', 'Total Correct', 'Visually Grounded', 'Prior Correct', 'Lucky Correct', '% Visually Grounded']))

for m in MODELS:
    if m not in all_model_stats: continue
    rows = all_model_stats[m]['ro
    correct = [r for r in rows if r['correct'].strip().lower() == 'true']
    vg  = sum(1 for r in correct if r['response_mode'] == 'VISUAL_GROUNDED')
    pc  = sum(1 for r in correct if r['response_mode'] == 'PRIOR_CORRECT')
    lc  = sum(1 for r in correct if r['response_mode'] == 'LUCKY_CORRECT')
    pct_vg = round(vg/len(correct)*100, 1) if correct else 0
    print('\t'.join([MODEL_LABELS[m], str(len(correct)), str(vg), str(pc), str(lc), str(pct_vg)+'%']))

print()
print('Done.')
