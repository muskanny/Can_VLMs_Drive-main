"""
eval_phase2_aw.py
-----------------
Phase 2 inference for Adverse Weather category.
Runs on a small targeted subset with a verbose prompt.
Models give up to 2-3 sentences of evidence-based reasoning.

Prompt format:
  Answer: Yes / No / Unanswerable
  Evidence: <1-3 sentences grounded in visible evidence>

Usage:
  python eval_phase2_aw.py --model paligemma
  python eval_phase2_aw.py --model smolvlm
  python eval_phase2_aw.py --model llava_ov
  python eval_phase2_aw.py --model internvl3

Output:
  /home2/muskan.singh/results/phase2_aw_<model>.csv
"""

import argparse, csv, json, os, time, re
import torch
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────
MANIFEST   = '/home2/muskan.singh/benchmark/junctions/ji_phase2_manifest.json'
IMG_BASE   = '/home2/muskan.singh/benchmark'
RESULTS    = '/home2/muskan.singh/results'
HF_CACHE   = '/home2/muskan.singh/hf_cache'

os.environ['HF_HOME']            = HF_CACHE
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE

PHASE2_PROMPT = (
    "Look at this driving scene and answer the following question.\n"
    "Question: {question}\n"
    "Answer yes or no, then briefly explain your reasoning in one or two sentences "
    "based only on what you can see in the image."
)

PHASE2_PROMPT_NOIMAGE = (
    "No image is provided.\n"
    "Question: {question}\n"
    "Can this question be answered without seeing an image? "
    "Answer yes or no, then explain in one sentence."
)

FIELDNAMES = [
    'category','image_id','q_id','q_type','question','ground_truth',
    'weather','model','mode','use_for','variation_type','source_q_id',
    'full_response','extracted_ans','correct','response_time'
]

# ── Answer extraction ───────────────────────────────────────────────────────
def extract_answer(response):
    """Extract yes/no/unanswerable from verbose response."""
    cleaned = response.replace('**','').strip()
    # Look for "Answer: X" pattern first
    m = re.search(r'Answer\s*:\s*(Yes|No|Unanswerable)', cleaned, re.IGNORECASE)
    if m:
        val = m.group(1).lower()
        if val == 'unanswerable': return 'unclear'
        return val
    # Fallback: startswith
    low = cleaned.lower()
    if low.startswith('yes'): return 'yes'
    if low.startswith('no'):  return 'no'
    return 'unclear'

# ── Model loaders ───────────────────────────────────────────────────────────
def load_paligemma():
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    model_id = 'google/paligemma2-3b-mix-224'
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={'': 0})
    model.eval()
    def infer(image, prompt):
        full_prompt = '<image>\n' + prompt
        inputs = processor(text=full_prompt, images=image, return_tensors='pt').to(model.device)
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return infer

def load_smolvlm():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model_id = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={'': 0})
    model.eval()
    def infer(image, prompt):
        messages = [{'role':'user','content':[
            {'type':'image'},{'type':'text','text':prompt}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if text is None:
            text = prompt
        inputs = processor(text=text, images=[image], return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
        # strip prompt echo — take content after last Assistant marker
        for marker in ['Assistant:', 'ASSISTANT:', '\nAnswer:']:
            if marker in decoded:
                decoded = decoded.split(marker)[-1].strip()
                break
        return decoded

def load_llava_ov():
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    model_id = 'lmms-lab/LLaVA-OneVision-1.5-8B-Instruct'
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True)
    model.eval()
    def infer(image, prompt):
        messages = [{'role':'user','content':[
            {'type':'image','image':image},
            {'type':'text','text':prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                          return_tensors='pt').to(model.device)
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return infer

def load_internvl3():
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    model_id = 'OpenGVLab/InternVL3-8B'
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, quantization_config=bnb, device_map={'': 0},
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()

    MEAN = (0.485,0.456,0.406); STD = (0.229,0.224,0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize((448,448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(MEAN, STD)
    ])

    def infer(image, prompt):
        pixel = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
        gen_cfg = dict(max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = model.chat(tokenizer, pixel, prompt, gen_cfg)
        return response
    return infer

LOADERS = {
    'paligemma': load_paligemma,
    'smolvlm':   load_smolvlm,
    'llava_ov':  load_llava_ov,
    'internvl3': load_internvl3,
}

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    required=True, choices=list(LOADERS.keys()))
    parser.add_argument('--no-image', action='store_true', help='Mode C: blank image baseline')
    args = parser.parse_args()

    mode   = 'C' if args.no_image else 'A'
    suffix = '_noimage' if args.no_image else ''
    out_path = os.path.join(RESULTS, f'phase2_ji_{args.model}{suffix}.csv')

    manifest = json.load(open(MANIFEST))

    # Resume logic
    done = set()
    if os.path.exists(out_path):
        for r in csv.DictReader(open(out_path)):
            done.add((r['image_id'], r['q_id']))
        print(f'Resuming — {len(done)} rows already done')

    print(f'Loading {args.model}...')
    infer = LOADERS[args.model]()
    print('Model loaded.')

    blank_image = Image.new('RGB', (224,224), (0,0,0))

    with open(out_path, 'a', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        if len(done) == 0:
            writer.writeheader()

        for entry in manifest:
            img_id = entry['image_id']
            img_path = os.path.join(IMG_BASE, entry['image_path'])
            # resolve dataset-specific subdirs
            if not os.path.exists(img_path):
                dataset = entry.get('dataset', 'bdd')
                if dataset == 'nuscenes':
                    img_path = os.path.join(IMG_BASE, 'nuscenes', 'images', entry['image_id'])
                else:
                    img_path = os.path.join(IMG_BASE, 'junctions', 'images', 'bdd', entry['image_id'])

            if args.no_image:
                image = blank_image
            else:
                image = Image.open(img_path).convert('RGB')

            for q in entry['questions']:
                key = (img_id, q['q_id'])
                if key in done:
                    continue

                prompt_template = PHASE2_PROMPT_NOIMAGE if args.no_image else PHASE2_PROMPT
                prompt = prompt_template.format(question=q['question'])

                t0 = time.time()
                try:
                    response = infer(image, prompt)
                except Exception as e:
                    response = f'ERROR: {e}'
                elapsed = round(time.time() - t0, 2)

                extracted = extract_answer(response)
                correct   = int(extracted == q['gt']) if extracted != 'unclear' else ''

                writer.writerow({
                    'category':      'junctions',
                    'image_id':      img_id,
                    'q_id':          q['q_id'],
                    'q_type':        q['q_type'],
                    'question':      q['question'],
                    'ground_truth':  q['gt'],
                    'weather':       entry.get('weather_condition',''),
                    'model':         args.model,
                    'mode':          mode,
                    'use_for':       q.get('use_for',''),
                    'variation_type':q.get('variation_type','original'),
                    'source_q_id':   q.get('source_q_id',''),
                    'full_response': response,
                    'extracted_ans': extracted,
                    'correct':       correct,
                    'response_time': elapsed,
                })
                fout.flush()
                print(f'  {img_id[:20]}  {q["q_id"]}  [{extracted}]  gt={q["gt"]}  {elapsed}s')

    print(f'\nDone. Results at {out_path}')
    wc = sum(1 for _ in open(out_path)) - 1
    print(f'Total rows written: {wc}')

if __name__ == '__main__':
    main()
