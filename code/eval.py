import json
import os
import csv
import time
import argparse
import torch
from PIL import Image

MANIFEST_DIR = '/home2/muskan.singh/results'
RESULTS_DIR = '/home2/muskan.singh/results'
IMG_DIR = '/home2/muskan.singh/benchmark/adverse_weather'  # default, overridden by --img-dir

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model name e.g. moondream')
    parser.add_argument('--category', required=True, help='adverse_weather | junctions | human_behaviour | negation_bias')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images for testing')
    parser.add_argument('--manifest', type=str, default=None, help='Explicit manifest path (optional)')
    parser.add_argument('--no-image', action='store_true', help='Run text-prior baseline with blank image')
    parser.add_argument('--suffix', type=str, default='', help='Output filename suffix e.g. _v2')
    parser.add_argument('--img-dir', type=str, default=None, help='Base image directory override')
    return parser.parse_args()

def override_img_dir(args):
    global IMG_DIR
    if args.img_dir:
        IMG_DIR = args.img_dir

def load_manifest(category, manifest_path=None):
    if manifest_path:
        path = manifest_path
    else:
        path = os.path.join(MANIFEST_DIR, f'{category}_manifest.json')
    with open(path) as f:
        return json.load(f)

def get_done_keys(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row['image_id'], row['q_id']))
    return done

def extract_yes_no(response):
    r = response.lower().strip().replace('**', '')
    if r.startswith('yes'): return 'yes'
    if r.startswith('no'): return 'no'
    # Fallback: check first sentence for clear signal
    first = r.split('.')[0]
    neg = ['is not ','are not ','cannot ','not possible','not visible',
           'not enough','not clear','not present','does not ','do not ',
           'unable to','not sufficient','not possible']
    pos = ['this is a ','there is a ','there are ','is visible','is present',
           'appears to be','shows a ','is a highway','is a motorway',
           'is nighttime','is daytime','is raining','is snowing']
    for p in neg:
        if p in first: return 'no'
    for p in pos:
        if p in first: return 'yes'
    # Last resort: scan individual words
    for word in r.split():
        if word in ('yes', 'yeah', 'correct', 'true'): return 'yes'
        if word in ('no', 'nope', 'incorrect', 'false'): return 'no'
    return 'unclear'

def is_correct(extracted, ground_truth, q_type):
    if q_type == 'identify':
        return None  # qualitative only
    if ground_truth in ('skip', 'PLACEHOLDER'):
        return None
    return extracted == ground_truth

def load_moondream():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Moondream 2B...")
    tokenizer = AutoTokenizer.from_pretrained(
        'vikhyatk/moondream2', revision='2024-08-26', trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        'vikhyatk/moondream2', revision='2024-08-26',
        trust_remote_code=True, torch_dtype=torch.float16,
        device_map={'': 0}
    )
    model.eval()
    def query(image, question):
        enc = model.encode_image(image)
        return model.answer_question(enc, question, tokenizer)
    return query

# def load_gemma():
#     import torch
#     from transformers import AutoProcessor, AutoModelForCausalLM

#     print("Loading Gemma 3 4B...")

#     model_id = "google/gemma-3-4b-it"

#     processor = AutoProcessor.from_pretrained(
#         model_id,
#         local_files_only=True
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.bfloat16,
#         device_map={"": 0},
#         local_files_only=True   # 🔥 IMPORTANT
#     )

#     model.eval()

#     def query(image, question):
#         prompt = f"<image>\n{question}\nAnswer with yes or no only."

#         inputs = processor(
#             text=prompt,
#             images=image,
#             return_tensors="pt"
#         ).to(model.device)

#         with torch.no_grad():
#             out = model.generate(**inputs, max_new_tokens=50)

#         return processor.decode(out[0], skip_special_tokens=True).split("\n")[-1].strip()

#     return query

def load_gemma():
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    print("Loading Gemma 3 4B...")
    processor = AutoProcessor.from_pretrained('google/gemma-3-4b-it')
    model = Gemma3ForConditionalGeneration.from_pretrained(
        'google/gemma-3-4b-it', torch_dtype=torch.bfloat16, device_map={'': 0}
    )
    model.eval()
    def query(image, question):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question + "\nAnswer with yes or no only."}
        ]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()
    return query

def load_qwen3():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    print("Loading Qwen3-VL 8B...")
    processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-8B-Instruct', torch_dtype=torch.float16, device_map={'': 0}
    )
    model.eval()
    def query(image, question):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': question + '\nAnswer with yes or no only.'}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=inputs, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return query

def load_internvl():
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    print("Loading InternVL3 8B...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModel.from_pretrained(
        'OpenGVLab/InternVL3-8B', quantization_config=bnb,
        torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True)
    model.eval()
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    def query(image, question):
        pixel_values = transform(image.convert('RGB')).unsqueeze(0).to(torch.bfloat16).to(model.device)
        prompt = f'<image>\n{question}\nAnswer with yes or no only.'
        generation_config = dict(max_new_tokens=50, do_sample=False)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()
    return query

def load_llava_ov():
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info
    print("Loading LLaVA-OneVision 1.5 8B...")
    processor = AutoProcessor.from_pretrained('lmms-lab/LLaVA-OneVision-1.5-8B-Instruct', trust_remote_code=True)
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        'lmms-lab/LLaVA-OneVision-1.5-8B-Instruct',
        quantization_config=bnb, device_map={'': 0}, trust_remote_code=True
    )
    model.eval()
    def query(image, question):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': question + '\nAnswer with yes or no only.'}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors='pt').to(model.device)
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return query


def load_smolvlm():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print("Loading SmolVLM2 2.2B...")
    processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-2.2B-Instruct')
    model = AutoModelForImageTextToText.from_pretrained(
        'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
        torch_dtype=torch.bfloat16, device_map={'': 0}
    )
    model.eval()
    def query(image, question):
        messages = [{'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': question + '\nAnswer with yes or no only.'}
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        decoded = processor.decode(out[0], skip_special_tokens=True)
        # Strip 'Assistant:' prefix if present
        decoded = decoded.split('\n')[-1].strip()
        decoded = decoded.replace('Assistant:', '').strip().rstrip('.')
        return decoded
    return query

def load_kimi():
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("Loading Kimi-VL-A3B Thinking...")
    processor = AutoProcessor.from_pretrained(
        'moonshotai/Kimi-VL-A3B-Thinking', trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        'moonshotai/Kimi-VL-A3B-Thinking',
        torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True
    )
    model.eval()
    def query(image, question):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': question + '\nAnswer with yes or no only.'}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)
        full = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        # Kimi thinking models wrap answer in <think>...</think> then give final answer
        if '</think>' in full:
            return full.split('</think>')[-1].strip()
        return full.strip()
    return query

def load_paligemma():
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    print("Loading PaliGemma 2 3B...")
    processor = AutoProcessor.from_pretrained('google/paligemma2-3b-mix-224')
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        'google/paligemma2-3b-mix-224',
        torch_dtype=torch.bfloat16, device_map={'': 0}
    )
    model.eval()
    def query(image, question):
        prompt = '<image>\n' + question + '\nAnswer with yes or no only.'
        inputs = processor(text=prompt, images=image, return_tensors='pt').to(model.device)
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return query

def load_qwen3_thinking():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    print("Loading Qwen3-VL 8B Thinking...")
    processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Thinking')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-8B-Thinking', torch_dtype=torch.float16, device_map={'': 0}
    )
    model.eval()
    def query(image, question):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': question + '\nAnswer with yes or no only.'}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=inputs, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)
        full = processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if '</think>' in full:
            return full.split('</think>')[-1].strip()
        return full.strip()
    return query

MODEL_LOADERS = {
    'moondream': load_moondream,
    'smolvlm': load_smolvlm,
    'kimi': load_kimi,
    'paligemma': load_paligemma,
    'gemma': load_gemma,
    'qwen3': load_qwen3,
    'qwen3_thinking': load_qwen3_thinking,
    'internvl3': load_internvl,
    'llava_ov': load_llava_ov,
}

def main():
    args = get_args()
    override_img_dir(args)
    assert args.model in MODEL_LOADERS, f"Unknown model. Choose from: {list(MODEL_LOADERS.keys())}"

    manifest = load_manifest(args.category, args.manifest)
    if args.limit:
        manifest = manifest[:args.limit]

    suffix = ('_noimage' if args.no_image else '') + args.suffix
    csv_path = os.path.join(RESULTS_DIR, f'{args.category}_{args.model}{suffix}.csv')
    done = get_done_keys(csv_path)
    print(f"Resuming from {len(done)} completed rows")

    query_fn = MODEL_LOADERS[args.model]()

    fieldnames = ['category', 'image_id', 'q_id', 'q_type', 'question',
                  'ground_truth', 'gt_tier', 'gt_confidence', 'gt_source',
                  'use_for', 'is_augmented', 'source_image_id',
                  'weather', 'model', 'full_response', 'extracted_ans',
                  'correct', 'response_time']

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done:
            writer.writeheader()

        for entry in manifest:
            image_id = entry['image_id']
            image_path = os.path.join(IMG_DIR, entry.get('image_path', os.path.join('images/bdd', image_id)))

            if args.no_image:
                image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            else:
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"  Skipping {image_id}: {e}")
                    continue

            for q in entry['questions']:
                key = (image_id, q['q_id'])
                if key in done:
                    continue
                if q.get('ground_truth') == 'PLACEHOLDER':
                    continue  # skip unannotated hard questions
                if q.get('gt') in ('PLACEHOLDER', None):
                    continue  # skip unannotated hard questions

                t0 = time.time()
                try:
                    response = query_fn(image, q['question'])
                except Exception as e:
                    response = f"ERROR: {e}"
                elapsed = round(time.time() - t0, 2)

                extracted = extract_yes_no(response)
                correct = is_correct(extracted, q.get('gt', ''), q.get('q_type', 'yes/no'))

                writer.writerow({
                    'category': args.category,
                    'image_id': image_id,
                    'q_id': q['q_id'],
                    'q_type': q.get('q_type', 'yes/no'),
                    'question': q['question'],
                    'ground_truth': q.get('gt', ''),
                    'gt_tier': q.get('gt_tier', ''),
                    'gt_confidence': q.get('gt_confidence', ''),
                    'gt_source': q.get('gt_source', 'automatic'),
                    'use_for': q.get('use_for', ''),
                    'is_augmented': entry.get('is_augmented', False),
                    'source_image_id': entry.get('source_image_id', ''),
                    'weather': entry.get('weather_condition', entry.get('weather', '')),
                    'model': args.model,
                    'full_response': response,
                    'extracted_ans': extracted,
                    'correct': correct,
                    'response_time': elapsed,
                })
                f.flush()
                done.add(key)

            print(f"  Done: {image_id}")

    print(f"\nFinished. Results saved to {csv_path}")

if __name__ == '__main__':
    main()