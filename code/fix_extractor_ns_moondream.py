import csv
import re

INPUT  = '/home2/muskan.singh/results/nuscenes_moondream_ns.csv'
OUTPUT = '/home2/muskan.singh/results/nuscenes_moondream_ns_fixed2.csv'

NO_SIGNALS  = ['is not', 'there is no', 'there are no', 'would not', 'cannot',
               'no vehicle', 'no pedestrian', 'not visible', 'not present',
               'no person', 'no one', 'nobody', 'not detected', 'does not']
YES_SIGNALS = ['there is a', 'there are', 'i can see', 'is visible', 'is present',
               'is approaching', 'there is one', 'is crossing', 'is in the']

def parse_response(raw):
    cleaned = raw.replace('**', '').strip()
    low = cleaned.lower()
    if low.startswith('yes'):
        return 'yes'
    if low.startswith('no'):
        return 'no'
    first_sent = re.split(r'[.!?]', low)[0]
    for sig in NO_SIGNALS:
        if sig in first_sent:
            return 'no'
    for sig in YES_SIGNALS:
        if sig in first_sent:
            return 'yes'
    excerpt = low[:200]
    for sig in NO_SIGNALS:
        if sig in excerpt:
            return 'no'
    for sig in YES_SIGNALS:
        if sig in excerpt:
            return 'yes'
    return 'unclear'

fixed = 0
total_unclear_in = 0
total_unclear_out = 0

with open(INPUT, newline='', encoding='utf-8') as fin,      open(OUTPUT, 'w', newline='', encoding='utf-8') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row['extracted_ans'] == 'unclear':
            total_unclear_in += 1
            new_ans = parse_response(row['full_response'])
            if new_ans != 'unclear':
                row['extracted_ans'] = new_ans
                row['correct'] = str(int(new_ans == row['ground_truth']))
                fixed += 1
            else:
                total_unclear_out += 1
        writer.writerow(row)

print(f"Input unclear:  {total_unclear_in}")
print(f"Resolved:       {fixed}")
print(f"Still unclear:  {total_unclear_out}")
print(f"Saved to:       {OUTPUT}")
