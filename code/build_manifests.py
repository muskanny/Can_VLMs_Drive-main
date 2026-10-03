import json
import os

ANN_FILE = '/home2/muskan.singh/val/annotations/bdd100k_labels_images_val.json'
IMG_DIR = '/home2/muskan.singh/val/images'
OUT_DIR = '/home2/muskan.singh/results'

ADVERSE = {'rainy', 'snowy', 'foggy'}
PRECIPITATION = {'rainy', 'snowy'}
HAZARDOUS = {'rainy', 'snowy', 'foggy'}
REDUCED_VIS = {'rainy', 'snowy', 'foggy'}

AW_QUESTIONS = {
    'AW_02': 'Is the horizon line clearly visible in this image?',
    'AW_03': 'Are lane markings visible for the full length of road shown in this image?',
    'AW_04r': 'Does the road surface show visible signs of wet or icy conditions?',
    'AW_05': 'What specific weather condition is affecting driving visibility in this scene?',
    'AW_07': 'Would a driver need to reduce speed due to visibility conditions in this scene?',
    'AW_09r': 'Is this image taken during daylight hours?',
    'AW_10': 'Can the ego vehicle safely maintain highway speed in these conditions?',
    'AW_11': 'Does the sky appearance indicate an ongoing weather event?',
    'AW_12r': 'Would a driver need to use headlights in these conditions?',
    'AW_N1': 'Is precipitation actively falling in this scene?',
    'AW_N2': 'Is this scene affected by reduced visibility due to atmospheric conditions?',
    'AW_N3': 'Would this weather condition be classified as hazardous for driving?',
}

JI_QUESTIONS = {
    'JI_01': 'Is there a traffic light visible in this scene?',
    'JI_02': 'Is this scene at an urban intersection?',
    'JI_03': 'Are there multiple vehicles visible at this junction?',
    'JI_04': 'Is this a highway scene with no traffic signals?',
    'JI_05': 'Are pedestrians present at this junction?',
    'JI_06': 'Is this scene in a residential area?',
    'JI_07': 'Are there more than two traffic lights visible in this scene?',
    'JI_08': 'Is this a nighttime junction scene?',
    'JI_09': 'Are there both pedestrians and traffic lights present in this scene?',
    'JI_10': 'Is there any bus or truck visible at this junction?',
}

HB_QUESTIONS = {
    'HB_01': 'Is there a person visible near the roadway in this scene?',
    'HB_02': 'Are there multiple people visible in this scene?',
    'HB_03': 'Are there more than five people visible in this scene?',
    'HB_04': 'Is there a cyclist or rider visible in this scene?',
    'HB_05': 'Are both pedestrians and vehicles present in this scene?',
    'HB_06': 'Is this a nighttime scene with pedestrians present?',
    'HB_07': 'Are there pedestrians present in adverse weather conditions?',
    'HB_08': 'Is there a bus or truck in the same scene as pedestrians?',
    'HB_09': 'Are there any pedestrians present on a highway scene?',
    'HB_10': 'Is this an urban scene with pedestrian activity?',
}

NB_QUESTIONS = {
    'NB_TL_01': ('traffic_light', 'absent', 'Is a traffic signal absent from this scene?'),
    'NB_TL_02': ('traffic_light', 'no', 'Is there no traffic light visible to the ego vehicle?'),
    'NB_TL_03': ('traffic_light', 'lack', 'Does this road lack any traffic signal control?'),
    'NB_TL_04': ('traffic_light', 'free of', 'Is this junction free of any overhead signals?'),
    'NB_TL_05': ('traffic_light', 'without', 'Is the intersection being managed without any lights?'),
    'NB_TL_06': ('traffic_light', 'without', 'Would a driver need to proceed without signal guidance here?'),
    'NB_PE_01': ('person', 'free of', 'Is the road ahead free of any pedestrians?'),
    'NB_PE_02': ('person', 'no', 'Is there no person visible near the carriageway?'),
    'NB_PE_03': ('person', 'no', 'Are there no humans walking or standing near the road?'),
    'NB_PE_04': ('person', 'lack', 'Does this scene lack any pedestrian activity?'),
    'NB_PE_05': ('person', 'empty of', 'Is the footpath empty of any individuals?'),
    'NB_PE_06': ('person', 'not', 'Would the driver not need to slow for any pedestrians here?'),
    'NB_LN_01': ('lane', 'without', 'Is this road completely without any lane markings?'),
    'NB_LN_02': ('lane', 'no', 'Are there no painted lines separating traffic on this road?'),
    'NB_LN_03': ('lane', 'lack', 'Does this road lack a centre dividing line?'),
    'NB_LN_04': ('lane', 'no', 'Is there no lane demarcation visible on the road surface?'),
    'NB_LN_05': ('lane', 'free of', 'Is this road free of any white or yellow painted lines?'),
    'NB_LN_06': ('lane', 'unable', 'Would a driver be unable to identify their lane from this image?'),
    'NB_CA_01': ('car', 'no', 'Is there no other vehicle visible on this road?'),
    'NB_CA_02': ('car', 'absent', 'Are other cars absent from this scene?'),
    'NB_CA_03': ('car', 'free of', 'Is this road free of any other moving vehicles?'),
    'NB_CA_04': ('car', 'lack', 'Does this scene lack any other road users?'),
    'NB_CA_05': ('car', 'without', 'Is the ego vehicle travelling without any other vehicles nearby?'),
    'NB_CA_06': ('car', 'unmarked', 'Is this road unmarked by the presence of other vehicles?'),
}


def get_categories(item):
    return set(l['category'] for l in item.get('labels', []) if 'category' in l)


def get_category_count(item, cat):
    return sum(1 for l in item.get('labels', []) if l.get('category') == cat)


def get_aw_gt(q_id, weather, timeofday):
    if q_id == 'AW_02':
        if weather == 'clear': return 'yes'
        if weather in ADVERSE: return 'no'
        return 'skip'
    if q_id == 'AW_03':
        if weather == 'clear': return 'yes'
        if weather in ADVERSE: return 'no'
        return 'skip'
    if q_id == 'AW_04r':
        if weather in PRECIPITATION: return 'yes'
        if weather == 'clear': return 'no'
        return 'skip'
    if q_id == 'AW_05':
        return weather
    if q_id == 'AW_07':
        if weather == 'clear': return 'no'
        if weather in ADVERSE: return 'yes'
        return 'skip'
    if q_id == 'AW_09r':
        if timeofday == 'daytime': return 'yes'
        if timeofday == 'night': return 'no'
        return 'skip'
    if q_id == 'AW_10':
        if weather == 'clear': return 'yes'
        if weather in ADVERSE: return 'no'
        return 'skip'
    if q_id == 'AW_11':
        if weather in PRECIPITATION: return 'yes'
        if weather in {'clear', 'foggy'}: return 'no'
        return 'skip'
    if q_id == 'AW_12r':
        if weather in ADVERSE or timeofday == 'night': return 'yes'
        if weather == 'clear' and timeofday == 'daytime': return 'no'
        return 'skip'
    if q_id == 'AW_N1':
        if weather in PRECIPITATION: return 'yes'
        if weather != 'undefined': return 'no'
        return 'skip'
    if q_id == 'AW_N2':
        if weather in REDUCED_VIS or timeofday == 'night': return 'yes'
        if weather == 'clear' and timeofday == 'daytime': return 'no'
        return 'skip'
    if q_id == 'AW_N3':
        if weather in HAZARDOUS: return 'yes'
        if weather == 'clear': return 'no'
        return 'skip'
    return 'skip'


def build_aw_manifest(data, max_per_condition=500):
    print("\nBuilding Adverse Weather manifest...")
    conditions = {'rainy': [], 'foggy': [], 'snowy': [], 'night': [], 'clear': []}
    for item in data:
        weather = item.get('attributes', {}).get('weather', 'undefined')
        timeofday = item.get('attributes', {}).get('timeofday', 'undefined')
        if weather == 'rainy' and len(conditions['rainy']) < max_per_condition:
            conditions['rainy'].append(item)
        elif weather == 'foggy' and len(conditions['foggy']) < max_per_condition:
            conditions['foggy'].append(item)
        elif weather == 'snowy' and len(conditions['snowy']) < max_per_condition:
            conditions['snowy'].append(item)
        elif timeofday == 'night' and weather not in ADVERSE and len(conditions['night']) < max_per_condition:
            conditions['night'].append(item)
        elif weather == 'clear' and timeofday == 'daytime' and len(conditions['clear']) < max_per_condition:
            conditions['clear'].append(item)

    selected = []
    for cond, items in conditions.items():
        print(f"  {cond}: {len(items)} images")
        selected.extend(items)

    manifest = []
    for item in selected:
        weather = item.get('attributes', {}).get('weather', 'undefined')
        timeofday = item.get('attributes', {}).get('timeofday', 'undefined')
        entry = {
            'image_id': item['name'],
            'image_path': os.path.join(IMG_DIR, item['name']),
            'weather': weather,
            'timeofday': timeofday,
            'questions': []
        }
        for q_id, q_text in AW_QUESTIONS.items():
            gt = get_aw_gt(q_id, weather, timeofday)
            entry['questions'].append({
                'q_id': q_id,
                'q_type': 'identify' if q_id == 'AW_05' else 'yes/no',
                'question': q_text,
                'ground_truth': gt
            })
        manifest.append(entry)

    out_path = os.path.join(OUT_DIR, 'adverse_weather_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved {len(manifest)} images to {out_path}")


def build_ji_manifest(data, max_imgs=2500):
    print("\nBuilding Junctions manifest...")
    selected = []
    for item in data:
        cats = get_categories(item)
        scene = item.get('attributes', {}).get('scene', '')
        if 'traffic light' in cats and scene == 'city street':
            selected.append(item)
    selected = selected[:max_imgs]
    print(f"  Selected {len(selected)} images")

    manifest = []
    for item in selected:
        cats = get_categories(item)
        scene = item.get('attributes', {}).get('scene', '')
        timeofday = item.get('attributes', {}).get('timeofday', 'undefined')
        tl_count = get_category_count(item, 'traffic light')
        car_count = get_category_count(item, 'car')
        person_present = 'person' in cats
        bus_truck = 'bus' in cats or 'truck' in cats

        gt_map = {
            'JI_01': 'yes' if 'traffic light' in cats else 'no',
            'JI_02': 'yes' if 'traffic light' in cats and scene == 'city street' else 'no',
            'JI_03': 'yes' if car_count >= 2 else 'no',
            'JI_04': 'yes' if scene == 'highway' and 'traffic light' not in cats else 'no',
            'JI_05': 'yes' if person_present else 'no',
            'JI_06': 'yes' if scene == 'residential' else 'no',
            'JI_07': 'yes' if tl_count > 2 else 'no',
            'JI_08': 'yes' if timeofday == 'night' else 'no',
            'JI_09': 'yes' if person_present and 'traffic light' in cats else 'no',
            'JI_10': 'yes' if bus_truck else 'no',
        }

        entry = {
            'image_id': item['name'],
            'image_path': os.path.join(IMG_DIR, item['name']),
            'scene': scene,
            'timeofday': timeofday,
            'questions': []
        }
        for q_id, q_text in JI_QUESTIONS.items():
            entry['questions'].append({
                'q_id': q_id,
                'q_type': 'yes/no',
                'question': q_text,
                'ground_truth': gt_map[q_id]
            })
        manifest.append(entry)

    out_path = os.path.join(OUT_DIR, 'junctions_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved {len(manifest)} images to {out_path}")


def build_hb_manifest(data, max_imgs=2500):
    print("\nBuilding Human Behaviour manifest...")
    person_present = [x for x in data if 'person' in get_categories(x)][:1250]
    person_absent = [x for x in data if 'person' not in get_categories(x)][:1250]
    selected = person_present + person_absent
    print(f"  person present: {len(person_present)}, absent: {len(person_absent)}")

    manifest = []
    for item in selected:
        cats = get_categories(item)
        scene = item.get('attributes', {}).get('scene', '')
        weather = item.get('attributes', {}).get('weather', 'undefined')
        timeofday = item.get('attributes', {}).get('timeofday', 'undefined')
        person_count = get_category_count(item, 'person')
        p = 'person' in cats
        car_present = 'car' in cats
        bus_truck = 'bus' in cats or 'truck' in cats
        rider_bike = 'rider' in cats or 'bike' in cats

        gt_map = {
            'HB_01': 'yes' if p else 'no',
            'HB_02': 'yes' if person_count >= 2 else 'no',
            'HB_03': 'yes' if person_count > 5 else 'no',
            'HB_04': 'yes' if rider_bike else 'no',
            'HB_05': 'yes' if p and car_present else 'no',
            'HB_06': 'yes' if timeofday == 'night' and p else 'no',
            'HB_07': 'yes' if p and weather in ADVERSE else 'no',
            'HB_08': 'yes' if p and bus_truck else 'no',
            'HB_09': 'yes' if scene == 'highway' and p else 'no',
            'HB_10': 'yes' if scene == 'city street' and p else 'no',
        }

        entry = {
            'image_id': item['name'],
            'image_path': os.path.join(IMG_DIR, item['name']),
            'scene': scene,
            'weather': weather,
            'timeofday': timeofday,
            'questions': []
        }
        for q_id, q_text in HB_QUESTIONS.items():
            entry['questions'].append({
                'q_id': q_id,
                'q_type': 'yes/no',
                'question': q_text,
                'ground_truth': gt_map[q_id]
            })
        manifest.append(entry)

    out_path = os.path.join(OUT_DIR, 'human_behaviour_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved {len(manifest)} images to {out_path}")


def build_nb_manifest(data, max_per_split=250):
    print("\nBuilding Negation Bias manifest...")
    obj_map = {
        'traffic_light': 'traffic light',
        'person': 'person',
        'lane': 'lane',
        'car': 'car'
    }

    all_entries = []
    for obj_key, obj_cat in obj_map.items():
        present = [x for x in data if obj_cat in get_categories(x)][:max_per_split]
        absent = [x for x in data if obj_cat not in get_categories(x)][:max_per_split]
        print(f"  {obj_key}: {len(present)} present, {len(absent)} absent")

        prefix = 'NB_' + obj_key[:2].upper()
        if obj_key == 'traffic_light':
            prefix = 'NB_TL'
        elif obj_key == 'person':
            prefix = 'NB_PE'
        elif obj_key == 'lane':
            prefix = 'NB_LN'
        elif obj_key == 'car':
            prefix = 'NB_CA'

        for split, items in [('present', present), ('absent', absent)]:
            for item in items:
                entry = {
                    'image_id': item['name'],
                    'image_path': os.path.join(IMG_DIR, item['name']),
                    'object': obj_key,
                    'gt_present': split == 'present',
                    'questions': []
                }
                for q_id, (obj, neg_word, q_text) in NB_QUESTIONS.items():
                    if not q_id.startswith(prefix):
                        continue
                    expected = 'no' if split == 'present' else 'yes'
                    entry['questions'].append({
                        'q_id': q_id,
                        'negation_word': neg_word,
                        'question': q_text,
                        'ground_truth': expected
                    })
                all_entries.append(entry)

    out_path = os.path.join(OUT_DIR, 'negation_bias_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(all_entries, f, indent=2)
    print(f"  Saved {len(all_entries)} entries to {out_path}")


def main():
    print("Loading BDD100K annotations...")
    with open(ANN_FILE) as f:
        data = json.load(f)
    print(f"Total images: {len(data)}")

    build_aw_manifest(data)
    build_ji_manifest(data)
    build_hb_manifest(data)
    build_nb_manifest(data)
    print("\nAll manifests built successfully.")


if __name__ == '__main__':
    main()