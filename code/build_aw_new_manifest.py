import json
import os

SELECTED_PATH   = '/home2/muskan.singh/results/aw_selected.json'
AUGMENTED_PATH  = '/home2/muskan.singh/results/aw_augmented_manifest.json'
OUTPUT_PATH     = '/home2/muskan.singh/results/aw_new_manifest.json'

# ── questions ──────────────────────────────────────────────────────────────
# Easy/medium tier — GT derivable from weather tag
QUESTIONS_EASY = [
    {'q_id': 'AW_02', 'q_type': 'yes/no', 'question': 'Is the horizon line clearly visible in this image?'},
    {'q_id': 'AW_03', 'q_type': 'yes/no', 'question': 'Are lane markings visible for the full length of road shown in this image?'},
    {'q_id': 'AW_04r', 'q_type': 'yes/no', 'question': 'Does the road surface show visible signs of wet or icy conditions?'},
    {'q_id': 'AW_05', 'q_type': 'identify', 'question': 'What specific weather condition is affecting driving visibility in this scene?'},
    {'q_id': 'AW_07', 'q_type': 'yes/no', 'question': 'Would a driver need to reduce speed due to visibility conditions in this scene?'},
    {'q_id': 'AW_09r', 'q_type': 'yes/no', 'question': 'Is this image taken during daylight hours?'},
    {'q_id': 'AW_10', 'q_type': 'yes/no', 'question': 'Can the ego vehicle safely maintain highway speed in these conditions?'},
    {'q_id': 'AW_11', 'q_type': 'yes/no', 'question': 'Does the sky appearance indicate an ongoing weather event?'},
    {'q_id': 'AW_12r', 'q_type': 'yes/no', 'question': 'Would a driver need to use headlights in these conditions?'},
    {'q_id': 'AW_N1', 'q_type': 'yes/no', 'question': 'Is precipitation actively falling in this scene?'},
    {'q_id': 'AW_N2', 'q_type': 'yes/no', 'question': 'Is this scene affected by reduced visibility due to atmospheric conditions?'},
    {'q_id': 'AW_N3', 'q_type': 'yes/no', 'question': 'Would this weather condition be classified as hazardous for driving?'},
]

# Hard tier — GT = 'PLACEHOLDER', filled in by human annotation
QUESTIONS_HARD = [
    {'q_id': 'AW_H1', 'q_type': 'free_form', 'question': 'At what approximate distance ahead would a driver first be able to detect a stopped vehicle in these conditions?'},
    {'q_id': 'AW_H2', 'q_type': 'yes/no',   'question': 'Is the weather severe enough to recommend pulling over and stopping rather than continuing to drive?'},
    {'q_id': 'AW_H3', 'q_type': 'free_form', 'question': 'Which road user type faces the highest risk in the current weather conditions visible in this scene?'},
    {'q_id': 'AW_H4', 'q_type': 'yes/no',   'question': 'Based on the road surface and sky conditions visible, has this weather event recently started rather than been ongoing for some time?'},
    {'q_id': 'AW_H5', 'q_type': 'yes/no',   'question': 'Would the braking distance for a vehicle travelling at 60km/h be significantly increased in these conditions?'},
    {'q_id': 'AW_H6', 'q_type': 'yes/no',   'question': 'Is the current visibility sufficient to safely execute an overtaking manoeuvre on this road?'},
]

# ── GT logic for easy questions ────────────────────────────────────────────
def get_easy_gt(q_id, weather, timeofday):
    is_bad_weather = weather in ('rainy', 'foggy', 'snowy')
    is_day = timeofday in ('daytime',)

    gt_map = {
        'AW_02':  'no'  if is_bad_weather else 'yes',   # horizon visible
        'AW_03':  'no'  if is_bad_weather else 'yes',   # lane markings visible full length
        'AW_04r': 'yes' if weather in ('rainy', 'snowy') else 'no',  # wet/icy road
        'AW_05':  weather,                               # identify weather
        'AW_07':  'yes' if is_bad_weather else 'no',    # reduce speed
        'AW_09r': 'yes' if is_day else 'no',            # daylight
        'AW_10':  'no'  if is_bad_weather else 'yes',   # maintain highway speed
        'AW_11':  'yes' if is_bad_weather else 'no',    # ongoing weather event
        'AW_12r': 'yes' if is_bad_weather else 'no',    # use headlights
        'AW_N1':  'yes' if weather in ('rainy', 'snowy') else 'no',  # precipitation falling
        'AW_N2':  'yes' if is_bad_weather else 'no',    # reduced visibility
        'AW_N3':  'yes' if is_bad_weather else 'no',    # hazardous
    }
    return gt_map.get(q_id, 'unknown')

def build_questions(weather, timeofday, annotate_hard=False):
    questions = []
    for q in QUESTIONS_EASY:
        questions.append({
            **q,
            'ground_truth': get_easy_gt(q['q_id'], weather, timeofday),
            'gt_source': 'automatic'
        })
    for q in QUESTIONS_HARD:
        questions.append({
            **q,
            'ground_truth': 'PLACEHOLDER',
            'gt_source': 'human_needed',
            'annotate': annotate_hard
        })
    return questions

# ── load data ──────────────────────────────────────────────────────────────
print("Loading selected images...")
with open(SELECTED_PATH) as f:
    selected = json.load(f)

print("Checking for augmented manifest...")
augmented = []
if os.path.exists(AUGMENTED_PATH):
    with open(AUGMENTED_PATH) as f:
        augmented = json.load(f)
    print(f"  Found {len(augmented)} augmented images")
else:
    print("  Augmented manifest not found yet — building with real images only")

# ── build manifest entries ─────────────────────────────────────────────────
manifest = []

# Real weather images — annotate hard questions on these (smaller set)
for split in ['real_rainy', 'real_snowy', 'real_foggy']:
    images = selected[split]
    for item in images:
        manifest.append({
            'image_id':   item['image_id'],
            'image_path': item['image_path'],
            'weather':    item['weather'],
            'timeofday':  item['timeofday'],
            'is_augmented': False,
            'source': 'bdd100k_real',
            'questions':  build_questions(item['weather'], item['timeofday'], annotate_hard=True)
        })

# Augmented images — placeholder GT for hard questions
for item in augmented:
    manifest.append({
        'image_id':          item['image_id'],
        'image_path':        item['image_path'],
        'weather':           item['weather'],
        'timeofday':         item.get('timeofday', 'daytime'),
        'is_augmented':      True,
        'source_image_id':   item['source_image_id'],
        'source':            'sd2_img2img',
        'aug_strength':      item.get('aug_strength'),
        'questions':         build_questions(item['weather'], item.get('timeofday', 'daytime'), annotate_hard=False)
    })

# ── summary ────────────────────────────────────────────────────────────────
real_count = sum(1 for e in manifest if not e['is_augmented'])
aug_count  = sum(1 for e in manifest if e['is_augmented'])
total_pairs = len(manifest) * len(QUESTIONS_EASY + QUESTIONS_HARD)
hard_to_annotate = sum(1 for e in manifest if not e['is_augmented'])

print(f"\nManifest summary:")
print(f"  Real images:      {real_count}")
print(f"  Augmented images: {aug_count}")
print(f"  Total images:     {len(manifest)}")
print(f"  Questions each:   {len(QUESTIONS_EASY + QUESTIONS_HARD)} (12 easy + 6 hard)")
print(f"  Total QA pairs:   {total_pairs}")
print(f"  Hard Q annotation needed for: {hard_to_annotate} images ({hard_to_annotate * len(QUESTIONS_HARD)} pairs)")

with open(OUTPUT_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
