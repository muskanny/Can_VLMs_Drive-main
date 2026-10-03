import json
import random
import os

random.seed(42)

MANIFEST_PATH = '/home2/muskan.singh/results/adverse_weather_manifest.json'
OUTPUT_PATH = '/home2/muskan.singh/results/aw_selected.json'

print("Loading existing manifest...")
with open(MANIFEST_PATH) as f:
    data = json.load(f)

buckets = {}
for item in data:
    w = item['weather']
    buckets.setdefault(w, []).append(item)

for w, items in buckets.items():
    print(f"  {w}: {len(items)} images")

rainy_selected = random.sample(buckets.get('rainy', []), min(100, len(buckets.get('rainy', []))))
snowy_selected = random.sample(buckets.get('snowy', []), min(100, len(buckets.get('snowy', []))))
foggy_selected = buckets.get('foggy', [])

clear_daytime = [x for x in buckets.get('clear', []) if x['timeofday'] == 'daytime']
print(f"\nClear daytime available: {len(clear_daytime)}")
clear_selected = random.sample(clear_daytime, min(100, len(clear_daytime)))

for item in clear_selected:
    item['augmentation_target'] = True
    item['aug_weather_targets'] = ['rainy', 'foggy', 'snowy']

selected = {
    'real_rainy': rainy_selected,
    'real_snowy': snowy_selected,
    'real_foggy': foggy_selected,
    'clear_for_augmentation': clear_selected
}

total_real = len(rainy_selected) + len(snowy_selected) + len(foggy_selected)
total_aug = len(clear_selected) * 3
print(f"\nSelected:")
print(f"  Real rainy:  {len(rainy_selected)}")
print(f"  Real snowy:  {len(snowy_selected)}")
print(f"  Real foggy:  {len(foggy_selected)}")
print(f"  Clear (aug source): {len(clear_selected)} → {total_aug} augmented images")
print(f"  Total after augmentation: {total_real + total_aug} images")

with open(OUTPUT_PATH, 'w') as f:
    json.dump(selected, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
