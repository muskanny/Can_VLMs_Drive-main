#!/usr/bin/env python3
"""
VLM Driving Benchmark — Comprehensive Analysis Script
Joins manifest metadata with CSV results for deep per-category breakdowns.

Usage:
    python analyze_results.py
    python analyze_results.py --results_dir /home2/muskan.singh/results/
    python analyze_results.py --model moondream
    python analyze_results.py --category negation_bias
"""

import os
import csv
import json
import argparse
from collections import defaultdict

RESULTS_DIR = '/home2/muskan.singh/results/'

# Negation word mapping from q_id
NB_NEGATION_WORDS = {
    'NB_TL_01': 'absent',   'NB_TL_02': 'no',       'NB_TL_03': 'lack',
    'NB_TL_04': 'free of',  'NB_TL_05': 'without',  'NB_TL_06': 'without',
    'NB_PE_01': 'free of',  'NB_PE_02': 'no',        'NB_PE_03': 'no',
    'NB_PE_04': 'lack',     'NB_PE_05': 'empty of',  'NB_PE_06': 'not',
    'NB_LN_01': 'without',  'NB_LN_02': 'no',        'NB_LN_03': 'lack',
    'NB_LN_04': 'no',       'NB_LN_05': 'free of',   'NB_LN_06': 'unable',
    'NB_CA_01': 'no',       'NB_CA_02': 'absent',    'NB_CA_03': 'free of',
    'NB_CA_04': 'lack',     'NB_CA_05': 'without',   'NB_CA_06': 'unmarked',
}

def sep(title='', char='=', width=70):
    if title:
        print(f"\n{'='*width}")
        print(f"  {title}")
        print(f"{'='*width}")
    else:
        print('  ' + '-' * (width-2))

def pct(correct, total):
    if total == 0: return 'N/A'
    return f"{correct/total*100:.1f}%"

def acc(rows):
    if not rows: return None
    correct = sum(1 for r in rows if r['correct'].strip().lower() == 'true')
    return correct / len(rows)

def load_csv(filepath):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['category'] == 'category':
                continue
            rows.append(row)
    return rows

def load_manifest(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {}
    with open(filepath) as f:
        data = json.load(f)
    return {entry['image_id']: entry for entry in data}

def load_all(results_dir, model_filter=None):
    all_rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.csv'):
            continue
        rows = load_csv(os.path.join(results_dir, fname))
        if model_filter:
            rows = [r for r in rows if r['model'] == model_filter]
        all_rows.extend(rows)
        print(f"  Loaded {len(rows):>6} rows from {fname}")
    return all_rows

def get_models(rows):
    return sorted(set(r['model'] for r in rows))

def get_categories(rows):
    order = ['adverse_weather', 'junctions', 'human_behaviour', 'negation_bias']
    present = set(r['category'] for r in rows)
    return [c for c in order if c in present] + [c for c in present if c not in order]

# ─────────────────────────────────────────────────────────────────────────────
def section_overall(rows):
    sep("1. OVERALL ACCURACY PER MODEL")
    models = get_models(rows)
    categories = get_categories(rows)
    cat_labels = [c[:14] for c in categories]
    print(f"\n  {'Model':<22}", end='')
    for c in cat_labels:
        print(f"{c:>14}", end='')
    print(f"{'OVERALL':>14}")
    sep()
    for model in models:
        m_rows = [r for r in rows if r['model'] == model]
        print(f"  {model:<22}", end='')
        for cat in categories:
            cat_rows = [r for r in m_rows if r['category'] == cat]
            c = sum(r['correct'].strip().lower()=='true' for r in cat_rows)
            print(f"{pct(c, len(cat_rows)):>14}", end='')
        c = sum(r['correct'].strip().lower()=='true' for r in m_rows)
        print(f"{pct(c, len(m_rows)):>14}")
    print(f"\n  Total rows: {len(rows)}")

# ─────────────────────────────────────────────────────────────────────────────
def section_per_question(rows):
    sep("2. PER-QUESTION ACCURACY PER MODEL")
    models = get_models(rows)
    for cat in get_categories(rows):
        cat_rows = [r for r in rows if r['category'] == cat]
        if not cat_rows: continue
        qids = sorted(set(r['q_id'] for r in cat_rows))
        print(f"\n  [{cat.upper()}]")
        print(f"  {'Q_ID':<16}", end='')
        for m in models:
            print(f"{m[:13]:>14}", end='')
        print()
        sep()
        for qid in qids:
            print(f"  {qid:<16}", end='')
            for model in models:
                q_rows = [r for r in cat_rows if r['q_id'] == qid and r['model'] == model]
                if q_rows:
                    c = sum(r['correct'].strip().lower()=='true' for r in q_rows)
                    print(f"{pct(c, len(q_rows)):>14}", end='')
                else:
                    print(f"{'N/A':>14}", end='')
            print()

# ─────────────────────────────────────────────────────────────────────────────
def section_adverse_weather(rows, results_dir):
    sep("3. ADVERSE WEATHER — DEEP ANALYSIS")
    aw_rows = [r for r in rows if r['category'] == 'adverse_weather']
    if not aw_rows:
        print("  No adverse weather results found.")
        return
    models = get_models(aw_rows)
    manifest = load_manifest(os.path.join(results_dir, 'adverse_weather_manifest.json'))
    for r in aw_rows:
        meta = manifest.get(r['image_id'], {})
        r['_weather'] = meta.get('weather', 'unknown')
        r['_timeofday'] = meta.get('timeofday', 'unknown')

    for label, key, groups in [
        ('3a. By Weather Condition', '_weather',
         sorted(set(r['_weather'] for r in aw_rows if r['_weather'] != 'unknown'))),
        ('3b. By Time of Day', '_timeofday',
         sorted(set(r['_timeofday'] for r in aw_rows if r['_timeofday'] != 'unknown'))),
    ]:
        print(f"\n  {label}")
        print(f"  {'Group':<22}", end='')
        for m in models: print(f"{m[:13]:>14}", end='')
        print(f"{'Images':>10}")
        sep()
        for g in groups:
            g_rows = [r for r in aw_rows if r[key] == g]
            print(f"  {g:<22}", end='')
            for model in models:
                m_rows = [r for r in g_rows if r['model'] == model]
                c = sum(r['correct'].strip().lower()=='true' for r in m_rows)
                print(f"{pct(c, len(m_rows)):>14}", end='')
            imgs = len(set(r['image_id'] for r in g_rows if r['model'] == models[0]))
            print(f"{imgs:>10}")

    print(f"\n  3c. Adverse vs Clear")
    ADVERSE = {'rainy', 'snowy', 'foggy'}
    print(f"  {'Condition':<22}", end='')
    for m in models: print(f"{m[:13]:>14}", end='')
    print()
    sep()
    for label, wset in [('adverse (rain/snow/fog)', ADVERSE), ('clear', {'clear'})]:
        g_rows = [r for r in aw_rows if r['_weather'] in wset]
        print(f"  {label:<22}", end='')
        for model in models:
            m_rows = [r for r in g_rows if r['model'] == model]
            c = sum(r['correct'].strip().lower()=='true' for r in m_rows)
            print(f"{pct(c, len(m_rows)):>14}", end='')
        print()

# ─────────────────────────────────────────────────────────────────────────────
def section_junctions(rows, results_dir):
    sep("4. JUNCTIONS & INTERSECTIONS — DEEP ANALYSIS")
    ji_rows = [r for r in rows if r['category'] == 'junctions']
    if not ji_rows:
        print("  No junctions results found.")
        return
    models = get_models(ji_rows)
    manifest = load_manifest(os.path.join(results_dir, 'junctions_manifest.json'))
    for r in ji_rows:
        meta = manifest.get(r['image_id'], {})
        r['_scene'] = meta.get('scene', 'unknown')
        r['_timeofday'] = meta.get('timeofday', 'unknown')

    for label, key, groups in [
        ('4a. By Scene Type', '_scene',
         sorted(set(r['_scene'] for r in ji_rows if r['_scene'] != 'unknown'))),
        ('4b. By Time of Day', '_timeofday',
         sorted(set(r['_timeofday'] for r in ji_rows if r['_timeofday'] != 'unknown'))),
    ]:
        print(f"\n  {label}")
        print(f"  {'Group':<22}", end='')
        for m in models: print(f"{m[:13]:>14}", end='')
        print(f"{'Images':>10}")
        sep()
        for g in groups:
            g_rows = [r for r in ji_rows if r[key] == g]
            print(f"  {g:<22}", end='')
            for model in models:
                m_rows = [r for r in g_rows if r['model'] == model]
                c = sum(r['correct'].strip().lower()=='true' for r in m_rows)
                print(f"{pct(c, len(m_rows)):>14}", end='')
            imgs = len(set(r['image_id'] for r in g_rows if r['model'] == models[0]))
            print(f"{imgs:>10}")

# ─────────────────────────────────────────────────────────────────────────────
def section_human_behaviour(rows, results_dir):
    sep("5. HUMAN BEHAVIOUR — DEEP ANALYSIS")
    hb_rows = [r for r in rows if r['category'] == 'human_behaviour']
    if not hb_rows:
        print("  No human behaviour results found.")
        return
    models = get_models(hb_rows)
    manifest = load_manifest(os.path.join(results_dir, 'human_behaviour_manifest.json'))
    for r in hb_rows:
        meta = manifest.get(r['image_id'], {})
        r['_scene'] = meta.get('scene', 'unknown')
        r['_weather'] = meta.get('weather', 'unknown')
        r['_timeofday'] = meta.get('timeofday', 'unknown')

    for label, key, groups in [
        ('5a. By Scene Type', '_scene',
         sorted(set(r['_scene'] for r in hb_rows if r['_scene'] != 'unknown'))),
        ('5b. By Weather', '_weather',
         sorted(set(r['_weather'] for r in hb_rows if r['_weather'] != 'unknown'))),
        ('5c. By Time of Day', '_timeofday',
         sorted(set(r['_timeofday'] for r in hb_rows if r['_timeofday'] != 'unknown'))),
    ]:
        print(f"\n  {label}")
        print(f"  {'Group':<22}", end='')
        for m in models: print(f"{m[:13]:>14}", end='')
        print(f"{'Images':>10}")
        sep()
        for g in groups:
            g_rows = [r for r in hb_rows if r[key] == g]
            print(f"  {g:<22}", end='')
            for model in models:
                m_rows = [r for r in g_rows if r['model'] == model]
                c = sum(r['correct'].strip().lower()=='true' for r in m_rows)
                print(f"{pct(c, len(m_rows)):>14}", end='')
            imgs = len(set(r['image_id'] for r in g_rows if r['model'] == models[0]))
            print(f"{imgs:>10}")

# ─────────────────────────────────────────────────────────────────────────────
def section_negation_bias(rows, results_dir):
    sep("6. NEGATION BIAS — DEEP ANALYSIS")
    nb_rows = [r for r in rows if r['category'] == 'negation_bias']
    if not nb_rows:
        print("  No negation bias results found.")
        return
    models = get_models(nb_rows)

    for r in nb_rows:
        r['_negation_word'] = NB_NEGATION_WORDS.get(r['q_id'], 'unknown')

    # 6a. Present vs Absent gap
    print(f"\n  6a. Present vs Absent Accuracy (Core Bias Measurement)")
    print(f"  {'Model':<22} {'Present':>10} {'Absent':>10} {'Gap':>10} {'Bias':>20}")
    sep()
    for model in models:
        m_rows = [r for r in nb_rows if r['model'] == model]
        present_rows = [r for r in m_rows if r['ground_truth'].strip() == 'no']
        absent_rows  = [r for r in m_rows if r['ground_truth'].strip() == 'yes']
        p = acc(present_rows); a = acc(absent_rows)
        if p is None or a is None: continue
        gap = (p - a) * 100
        bias = "→ PRESENCE (strong)" if gap > 20 else \
               "→ presence (mild)"   if gap > 5  else \
               "→ ABSENCE (strong)"  if gap < -20 else \
               "→ absence (mild)"    if gap < -5  else "balanced"
        print(f"  {model:<22} {p*100:>9.1f}% {a*100:>9.1f}% {gap:>+9.1f}% {bias:>20}")

    # 6b. Per-object
    print(f"\n  6b. Per-Object Accuracy")
    obj_map = {'TL': 'Traffic Light', 'PE': 'Pedestrian', 'LN': 'Lane Marking', 'CA': 'Car'}
    for obj_code, obj_name in obj_map.items():
        obj_rows = [r for r in nb_rows if r['q_id'].startswith(f'NB_{obj_code}_')]
        if not obj_rows: continue
        print(f"\n  [{obj_name}]")
        print(f"  {'Model':<22} {'Overall':>10} {'Present':>10} {'Absent':>10}")
        sep()
        for model in models:
            m_rows = [r for r in obj_rows if r['model'] == model]
            if not m_rows: continue
            p_rows = [r for r in m_rows if r['ground_truth'].strip() == 'no']
            a_rows = [r for r in m_rows if r['ground_truth'].strip() == 'yes']
            c_all = sum(r['correct'].strip().lower()=='true' for r in m_rows)
            c_p   = sum(r['correct'].strip().lower()=='true' for r in p_rows)
            c_a   = sum(r['correct'].strip().lower()=='true' for r in a_rows)
            print(f"  {model:<22} {pct(c_all,len(m_rows)):>10} {pct(c_p,len(p_rows)):>10} {pct(c_a,len(a_rows)):>10}")

    # 6c. Per-negation-word
    print(f"\n  6c. Per-Negation-Word Accuracy")
    neg_words = sorted(set(r['_negation_word'] for r in nb_rows if r['_negation_word'] != 'unknown'))
    print(f"  {'Negation Word':<18}", end='')
    for m in models: print(f"{m[:13]:>14}", end='')
    print()
    sep()
    for word in neg_words:
        word_rows = [r for r in nb_rows if r['_negation_word'] == word]
        print(f"  {word:<18}", end='')
        for model in models:
            m_rows = [r for r in word_rows if r['model'] == model]
            if m_rows:
                c = sum(r['correct'].strip().lower()=='true' for r in m_rows)
                print(f"{pct(c, len(m_rows)):>14}", end='')
            else:
                print(f"{'N/A':>14}", end='')
        print()

    # 6d. Negation word present vs absent per model
    print(f"\n  6d. Negation Word — Present vs Absent Breakdown")
    for model in models:
        m_nb = [r for r in nb_rows if r['model'] == model]
        print(f"\n  Model: {model}")
        print(f"  {'Word':<18} {'Present':>10} {'Absent':>10} {'Gap':>10}")
        sep()
        for word in neg_words:
            w_rows = [r for r in m_nb if r['_negation_word'] == word]
            p_rows = [r for r in w_rows if r['ground_truth'].strip() == 'no']
            a_rows = [r for r in w_rows if r['ground_truth'].strip() == 'yes']
            p = acc(p_rows); a = acc(a_rows)
            if p is None or a is None: continue
            gap = (p - a) * 100
            print(f"  {word:<18} {p*100:>9.1f}% {a*100:>9.1f}% {gap:>+9.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
def section_response_quality(rows):
    sep("7. RESPONSE QUALITY — EXTRACTED ANSWER DISTRIBUTION")
    models = get_models(rows)
    for model in models:
        m_rows = [r for r in rows if r['model'] == model]
        total = len(m_rows)
        yes = sum(1 for r in m_rows if r['extracted_ans'].strip() == 'yes')
        no  = sum(1 for r in m_rows if r['extracted_ans'].strip() == 'no')
        unc = sum(1 for r in m_rows if r['extracted_ans'].strip() == 'unclear')
        print(f"\n  {model} (total={total})")
        print(f"    yes={yes} ({pct(yes,total)})  no={no} ({pct(no,total)})  unclear={unc} ({pct(unc,total)})")

# ─────────────────────────────────────────────────────────────────────────────
def section_response_time(rows):
    sep("8. AVERAGE RESPONSE TIME PER MODEL")
    models = get_models(rows)
    print(f"\n  {'Model':<22} {'Avg/query':>12} {'Total hrs':>12} {'Queries':>10}")
    sep()
    for model in models:
        times = []
        for r in [r for r in rows if r['model'] == model]:
            try: times.append(float(r['response_time']))
            except: pass
        if times:
            print(f"  {model:<22} {sum(times)/len(times):>11.2f}s {sum(times)/3600:>11.1f}h {len(times):>10}")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    parser.add_argument('--model', default=None)
    parser.add_argument('--category', default=None)
    args = parser.parse_args()

    print(f"\nLoading results from {args.results_dir}")
    all_rows = load_all(args.results_dir, model_filter=args.model)
    if args.category:
        all_rows = [r for r in all_rows if r['category'] == args.category]
    if not all_rows:
        print("No rows found.")
        return

    print(f"Models: {get_models(all_rows)}")
    print(f"Total rows: {len(all_rows)}")

    section_overall(all_rows)
    section_per_question(all_rows)
    section_adverse_weather(all_rows, args.results_dir)
    section_junctions(all_rows, args.results_dir)
    section_human_behaviour(all_rows, args.results_dir)
    section_negation_bias(all_rows, args.results_dir)
    section_response_quality(all_rows)
    section_response_time(all_rows)

    print(f"\n{'='*70}")
    print("  Analysis complete.")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
