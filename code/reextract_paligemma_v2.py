import csv, json, os
from collections import Counter

def extract_yes_no(response):
    r = response.lower().strip().replace('**', '')
    if r.startswith('yes'): return 'yes'
    if r.startswith('no'): return 'no'
    # PaliGemma descriptive single-word mappings
    yes_words = {'challenging','adverse','raining','rain','rainy','snowing',
                 'snow','snowy','foggy','fog','wet','night','nighttime',
                 'dark','reduced','limited','poor','slippery','icy'}
    no_words  = {'dry','clear','daytime','day','sunny','urban','residential',
                 'city','town','bright','good','normal','fine'}
    # Single word check
    words = r.split()
    if len(words) == 1:
        if words[0] in yes_words: return 'yes'
        if words[0] in no_words:  return 'no'
    # First sentence patterns
    first = r.split('.')[0]
    neg = ['is not ','are not ','cannot ','not possible','not visible',
           'not enough','not clear','not present','does not ','do not ',
           'unable to','not sufficient','unanswerable']
    pos = ['this is a ','there is a ','there are ','is visible','is present',
           'appears to be','shows a ','is a highway','is a motorway',
           'is nighttime','is daytime','is raining','is snowing']
    for p in neg:
        if p in first: return 'no'
    for p in pos:
        if p in first: return 'yes'
    for word in r.split():
        if word in ('yes','yeah','correct','true'): return 'yes'
        if word in ('no','nope','incorrect','false'): return 'no'
    return 'unclear'

def is_correct(extracted, ground_truth):
    if ground_truth in ('skip','PLACEHOLDER'): return ''
    if extracted == 'unclear': return 'False'
    return str(extracted == ground_truth)

with open('/home2/muskan.singh/aw_manifest_v2.json') as mf:
    manifest = json.load(mf)
weather_lookup = {e['image_id']: e.get('weather_condition','') for e in manifest}

for suffix, label in [('', 'Mode A'), ('_noimage', 'Mode C')]:
    INPUT  = f'/home2/muskan.singh/results/adverse_weather_paligemma_v2{suffix}.csv'
    BACKUP = f'/home2/muskan.singh/results/adverse_weather_paligemma_v2{suffix}_raw.csv'
    if not os.path.exists(INPUT):
        print(f'Skipping {label} — file not found')
        continue
    os.system(f'cp {INPUT} {BACKUP}')
    with open(INPUT, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    before = sum(1 for r in rows if r['extracted_ans']=='unclear')
    fixed = 0
    for r in rows:
        new_ans = extract_yes_no(r['full_response'])
        if new_ans != r['extracted_ans']: fixed += 1
        r['extracted_ans'] = new_ans
        r['correct'] = is_correct(new_ans, r['ground_truth'])
        if not r['weather']:
            r['weather'] = weather_lookup.get(r['image_id'], '')
    after = sum(1 for r in rows if r['extracted_ans']=='unclear')
    with open(INPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{label}: {len(rows)} rows | unclear {before} -> {after} | fixed {fixed}')
    print(f'  ans dist: {dict(Counter(r["extracted_ans"] for r in rows))}')
    print(f'  backed up to *_raw.csv')
