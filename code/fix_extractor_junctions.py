"""
fix_extractor_junctions.py
--------------------------
Post-hoc extractor fix for junctions inference CSVs.
Applies to all 5 models. Run BEFORE any analysis.

Fix rules (conservative):
1. Strip markdown bold (**) from full_response before parsing
2. For rows where extracted_ans == 'unclear':
   - Re-parse cleaned response with keyword scan fallback
   - Only flip to yes/no if a STRONG signal is found
   - Otherwise leave as unclear
3. Recalculate 'correct' field
4. Save as junctions_<model>_ji_v1_fixed.csv

Keyword logic (conservative):
  NO signals  : starts with "no" | "there is no" | "there are no" | "is not" |
                "are not" | "would not" | "cannot" | "can't" | "does not" |
                "don't" | "didn't" | "no," | "no." | "not visible" |
                "not present" | "not a" | "no pedestrian" | "no vehicle" |
                "no traffic" | "no signal" | "no marking"
  YES signals : starts with "yes" | "there is a" | "there are" | "i can see" |
                "the image shows" | "yes," | "yes." | "visible" (as first word)
  
  If both signals found → unclear (ambiguous response)
  If neither found → unclear (can't determine)
"""

import csv
import os
import re

RESULTS_DIR = "/home2/muskan.singh/results"

MODELS = [
    "moondream",
    "paligemma",
    "smolvlm",
    "llava_ov",
    "internvl3",
]

NO_SIGNALS = [
    "there is no ", "there are no ", "there's no ",
    "is not ", "are not ", "was not ", "were not ",
    "would not ", "will not ",
    "cannot ", "can't ", "cant ",
    "does not ", "do not ", "don't ", "didn't ",
    "not visible", "not present", "not a ",
    "no pedestrian", "no vehicle", "no traffic",
    "no signal", "no marking", "no lane",
    "no, ", "no.",
]

YES_SIGNALS = [
    "there is a ", "there are ", "there's a ",
    "i can see ", "i can confirm",
    "the image shows", "the scene shows",
    "yes, ", "yes.",
    "is present", "are present",
    "is visible", "are visible",
]

def clean_response(text):
    """Strip markdown bold/italic and extra whitespace."""
    text = text.replace("**", "").replace("*", "")
    text = text.strip()
    return text

def extract_conservative(text):
    """
    Conservative yes/no extraction.
    Returns 'yes', 'no', or 'unclear'.
    """
    t = clean_response(text).lower()

    # Direct startswith check first (strongest signal)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"

    # Keyword scan — look for NO signals
    no_found = any(sig in t for sig in NO_SIGNALS)
    # Keyword scan — look for YES signals
    yes_found = any(sig in t for sig in YES_SIGNALS)

    if no_found and not yes_found:
        return "no"
    if yes_found and not no_found:
        return "yes"

    # Both or neither → unclear
    return "unclear"

def fix_csv(model):
    input_path = os.path.join(RESULTS_DIR, f"junctions_{model}_ji_v1.csv")
    output_path = os.path.join(RESULTS_DIR, f"junctions_{model}_ji_v1_fixed.csv")

    if not os.path.exists(input_path):
        print(f"  [SKIP] {input_path} not found")
        return

    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    changed = 0
    still_unclear = 0
    bold_stripped = 0

    for row in rows:
        original_ans = row['extracted_ans']
        full_response = row.get('full_response', '')

        # Always re-parse if response contains ** (markdown bold)
        has_bold = '**' in full_response
        if has_bold:
            bold_stripped += 1

        # Re-parse if unclear OR has bold markdown
        if original_ans == 'unclear' or has_bold:
            new_ans = extract_conservative(full_response)

            if new_ans != original_ans:
                changed += 1
                row['extracted_ans'] = new_ans
                # Recalculate correct
                row['correct'] = str(
                    new_ans == row.get('ground_truth', '').lower()
                ).upper()  # TRUE / FALSE

        if row['extracted_ans'] == 'unclear':
            still_unclear += 1

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Model: {model}")
    print(f"  Total rows     : {total}")
    print(f"  Had bold (**)  : {bold_stripped}")
    print(f"  Changed        : {changed}")
    print(f"  Still unclear  : {still_unclear}")
    print(f"  Saved to       : {output_path}")

def main():
    print("=" * 60)
    print("Junctions Extractor Fix — Conservative Keyword Scan")
    print("=" * 60)
    for model in MODELS:
        fix_csv(model)
    print("\nDone. Use *_fixed.csv files for all analysis.")

if __name__ == "__main__":
    main()
