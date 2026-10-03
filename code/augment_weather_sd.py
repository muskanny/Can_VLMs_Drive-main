import json
import os
import torch
import time
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# ── paths ──────────────────────────────────────────────────────────────────
SELECTED_PATH   = '/home2/muskan.singh/results/aw_selected.json'
OUTPUT_DIR      = '/home2/muskan.singh/val/images_augmented/'
OUTPUT_MANIFEST = '/home2/muskan.singh/results/aw_augmented_manifest.json'
HF_CACHE        = '/home2/muskan.singh/hf_cache'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── weather configs — SD v1-5 img2img for rain/snow, skip fog for now ─────
WEATHER_CONFIGS = {
    'rainy': {
        'prompt': 'heavy rain on a city road, wet asphalt, rain streaks, overcast sky, puddles, realistic dashcam photo, photorealistic',
        'negative_prompt': 'sunny, clear sky, dry road, snow, fog, painting, illustration, artwork, cartoon, anime, oil painting, unrealistic, render',
        'strength': 0.30,
        'guidance_scale': 12.0,
    },
    'snowy': {
        'prompt': 'heavy snowfall on a road, snow on ground, snow covering road, winter driving, realistic dashcam photo',
        'negative_prompt': 'sunny, clear sky, dry road, rain, fog, summer',
        'strength': 0.30,
    },
    'foggy': {
        'prompt': 'foggy morning on a city road, dense mist, reduced visibility, grey hazy atmosphere, realistic dashcam photo, photorealistic',
        'negative_prompt': 'clear sky, sunny, bright, rain, snow, sharp visibility, painting, illustration, CGI, cinematic',
        'strength': 0.35,
        'guidance_scale': 10.0,
    },
}

# ── load model ─────────────────────────────────────────────────────────────
print("Loading SD v1-5 img2img pipeline...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    'SG161222/Realistic_Vision_V5.1_noVAE',
    torch_dtype=torch.float16,
    cache_dir=HF_CACHE,
)
pipe = pipe.to('cuda')
pipe.enable_attention_slicing()
print("Model loaded.")

# ── load selected images ───────────────────────────────────────────────────
with open(SELECTED_PATH) as f:
    selected = json.load(f)

clear_images = selected['clear_for_augmentation']
print(f"Augmenting {len(clear_images)} clear images × {len(WEATHER_CONFIGS)} weather types = {len(clear_images)*len(WEATHER_CONFIGS)} total")

# ── resume support ─────────────────────────────────────────────────────────
done = set()
if os.path.exists(OUTPUT_MANIFEST):
    with open(OUTPUT_MANIFEST) as f:
        existing = json.load(f)
    done = {(e['source_image_id'], e['aug_weather']) for e in existing}
    augmented_entries = existing
    print(f"Resuming — {len(done)} already done")
else:
    augmented_entries = []

# ── augmentation loop ──────────────────────────────────────────────────────
for i, item in enumerate(clear_images):
    src_path = item['image_path']
    src_id   = item['image_id']

    try:
        src_img = Image.open(src_path).convert('RGB').resize((768, 432))
    except Exception as e:
        print(f"  SKIP {src_id} — could not open: {e}")
        continue

    for weather, cfg in WEATHER_CONFIGS.items():
        if (src_id, weather) in done:
            continue

        out_filename = f"aug_{weather}_{src_id}"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        t0 = time.time()
        try:
            result = pipe(
                prompt=cfg['prompt'],
                negative_prompt=cfg['negative_prompt'],
                image=src_img,
                strength=cfg['strength'],
                guidance_scale=cfg.get('guidance_scale', 8.5),
                num_inference_steps=30,
            ).images[0]
            result.save(out_path)
            elapsed = round(time.time() - t0, 2)

            entry = {
                'image_id':          out_filename,
                'image_path':        out_path,
                'source_image_id':   src_id,
                'source_image_path': src_path,
                'weather':           weather,
                'aug_weather':       weather,
                'is_augmented':      True,
                'timeofday':         item['timeofday'],
                'aug_model':         'sd-v1-5-img2img',
                'aug_strength':      cfg['strength'],
                'aug_time_s':        elapsed,
            }
            augmented_entries.append(entry)
            done.add((src_id, weather))

            with open(OUTPUT_MANIFEST, 'w') as f:
                json.dump(augmented_entries, f, indent=2)

            print(f"  [{i+1}/{len(clear_images)}] {weather} {src_id} ({elapsed}s)")

        except Exception as e:
            print(f"  ERROR {src_id} {weather}: {e}")

print(f"\nDone. {len(augmented_entries)} augmented images saved.")
print(f"Manifest: {OUTPUT_MANIFEST}")
