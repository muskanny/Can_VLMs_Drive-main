import csv
import os

def extract_yes_no(response):
    r = response.lower().strip().replace('**', '')
    if r.startswith('yes'): return 'yes'
    if r.startswith('no'): return 'no'
    first = r.split('.')[0]
    neg = ['is not ','are not ','cannot ','not possible','not visible',
           'not enough','not clear','not present','does not ','do not ',
           'unable to','not sufficient']
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
    if ground_truth in ('skip', 'PLACEHOLDER'): return ''
    if extracted == 'unclear': return 'False'
    return str(extracted == ground_truth)

INPUT  = '/home2/muskan.singh/results/adverse_weather_moondream_v2.csv'
OUTPUT = '/home2/muskan.singh/results/adverse_weather_moondream_v2.csv'
BACKUP = '/home2/muskan.singh/results/adverse_weather_moondream_v2_raw.csv'

# Load manifest for weather backfill
import json
with open('/home2/muskan.singh/aw_manifest_v2.json') as mf:
    manifest = json.load(mf)
weather_lookup = {entry['image_id']: entry.get('weather_condition', '') for entry in manifest}

# Backup original first
os.system(f'cp {INPUT} {BACKUP}')
print(f'Backup saved to {BACKUP}')

with open(INPUT, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys())
before_unclear = sum(1 for r in rows if r['extracted_ans'] == 'unclear')

fixed = 0
for r in rows:
    new_ans = extract_yes_no(r['full_response'])
    if new_ans != r['extracted_ans']:
        fixed += 1
    r['extracted_ans'] = new_ans
    r['correct'] = is_correct(new_ans, r['ground_truth'])
    if not r['weather']:
        r['weather'] = weather_lookup.get(r['image_id'], '')

after_unclear = sum(1 for r in rows if r['extracted_ans'] == 'unclear')

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'Total rows: {len(rows)}')
print(f'Unclear before: {before_unclear} ({before_unclear/len(rows)*100:.1f}%)')
print(f'Unclear after:  {after_unclear} ({after_unclear/len(rows)*100:.1f}%)')
print(f'Rows changed: {fixed}')
print(f'Done. Original backed up as *_raw.csv')
