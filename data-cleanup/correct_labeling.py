import json
import random
from pathlib import Path
from collections import Counter

TRAIN_PATH = "data/train_.coco.json"
VAL_PATH   = "data/valid_.coco.json"
TEST_PATH  = "data/test_.coco.json"

OUT_DIR = Path(".")
MERGED_CLEAN_PATH = OUT_DIR / "data/merged_cleaned.coco.json"
TRAIN_OUT = OUT_DIR / "data/train_70.coco.json"
VAL_OUT   = OUT_DIR / "data/valid_15.coco.json"
TEST_OUT  = OUT_DIR / "data/test_15.coco.json"

SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
REINDEX_IDS_PER_SPLIT = True 

CRACK_SYNONYMS = {"crack", "crack ", " crack", "Crack", "CRACK", "Crank", "CRANK"}  # include the typo "Crank"
DROP_CATS = {"UnCracked", "uncracked", "objects", "Objects", "0"}

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def summarize_split(name, coco):
    imgs = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    counts = Counter([a["category_id"] for a in anns])
    per_cat = ", ".join(f"{cats.get(k, k)}={v}" for k, v in sorted(counts.items()))
    img_has_ann = {a["image_id"] for a in anns}
    zero_ann_images = sum(1 for img in imgs if img["id"] not in img_has_ann)
    print(f"[{name}] images={len(imgs)} | anns={len(anns)} | per-category: {per_cat or '—'} | zero-ann images={zero_ann_images}")

def merge_three(train_path, val_path, test_path):
    merged = None
    img_offset = ann_offset = 0
    all_images = []
    all_anns = []

    for fp in [train_path, val_path, test_path]:
        data = load_json(fp)
        if merged is None:
            merged = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "categories": data.get("categories", []),
                "images": [],
                "annotations": []
            }
        id_map = {}
        for img in data.get("images", []):
            new_id = img["id"] + img_offset
            id_map[img["id"]] = new_id
            new_img = dict(img); new_img["id"] = new_id
            all_images.append(new_img)
        for ann in data.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = ann["id"] + ann_offset
            new_ann["image_id"] = id_map[ann["image_id"]]
            all_anns.append(new_ann)
        if all_images:
            img_offset = max(img["id"] for img in all_images) + 1
        if all_anns:
            ann_offset = max(a["id"] for a in all_anns) + 1

    merged["images"] = all_images
    merged["annotations"] = all_anns
    return merged

def clean_to_binary_crack(coco):
    cats = coco.get("categories", [])
    name_by_id = {c["id"]: c["name"] for c in cats}

    action_by_id = {}
    for cid, name in name_by_id.items():
        n = (name or "").strip()
        if n in CRACK_SYNONYMS:
            action_by_id[cid] = "keep_crack"
        elif n in DROP_CATS:
            action_by_id[cid] = "drop"
        else:
            action_by_id[cid] = "drop"  

    new_anns = []
    new_id = 1
    for a in coco.get("annotations", []):
        act = action_by_id.get(a["category_id"], "drop")
        if act == "keep_crack":
            na = dict(a)
            na["id"] = new_id
            na["category_id"] = 1
            new_anns.append(na)
            new_id += 1

    new_cats = [{"id": 1, "name": "crack", "supercategory": "objects"}]
    cleaned = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": new_cats,
        "images": coco.get("images", []),
        "annotations": new_anns
    }
    return cleaned

def subset_coco(base, images, annotations, reindex=False):
    coco = {
        "info": base.get("info", {}),
        "licenses": base.get("licenses", []),
        "categories": base.get("categories", []),
        "images": [],
        "annotations": []
    }
    if not reindex:
        coco["images"] = images
        coco["annotations"] = annotations
        return coco

    img_id_map = {}
    new_images = []
    for new_id, img in enumerate(images, start=1):
        img_id_map[img["id"]] = new_id
        ni = dict(img); ni["id"] = new_id
        new_images.append(ni)
    new_annotations = []
    for new_ann_id, ann in enumerate(annotations, start=1):
        na = dict(ann)
        na["id"] = new_ann_id
        na["image_id"] = img_id_map[ann["image_id"]]
        new_annotations.append(na)
    coco["images"] = new_images
    coco["annotations"] = new_annotations
    return coco

def split_70_15_15(cleaned):
    random.seed(SEED)
    images = cleaned["images"][:]
    anns   = cleaned["annotations"]

    random.shuffle(images)
    N = len(images)
    n_train = int(N * TRAIN_RATIO)
    n_val   = int(N * VAL_RATIO)
    n_test  = N - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train+n_val]
    test_imgs  = images[n_train+n_val:]

    train_ids = {i["id"] for i in train_imgs}
    val_ids   = {i["id"] for i in val_imgs}
    test_ids  = {i["id"] for i in test_imgs}

    train_anns = [a for a in anns if a["image_id"] in train_ids]
    val_anns   = [a for a in anns if a["image_id"] in val_ids]
    test_anns  = [a for a in anns if a["image_id"] in test_ids]

    train_coco = subset_coco(cleaned, train_imgs, train_anns, reindex=REINDEX_IDS_PER_SPLIT)
    val_coco   = subset_coco(cleaned, val_imgs,   val_anns,   reindex=REINDEX_IDS_PER_SPLIT)
    test_coco  = subset_coco(cleaned, test_imgs,  test_anns,  reindex=REINDEX_IDS_PER_SPLIT)

    return train_coco, val_coco, test_coco

def main():
    merged = merge_three(TRAIN_PATH, VAL_PATH, TEST_PATH)

    cleaned = clean_to_binary_crack(merged)
    save_json(cleaned, MERGED_CLEAN_PATH)
    summarize_split("Merged-Clean", cleaned)
    print(f"Saved merged+cleaned: {MERGED_CLEAN_PATH}\n")

    train_coco, val_coco, test_coco = split_70_15_15(cleaned)

    save_json(train_coco, TRAIN_OUT)
    save_json(val_coco,   VAL_OUT)
    save_json(test_coco,  TEST_OUT)

    summarize_split("Train 70%", train_coco)
    summarize_split("Valid 15%", val_coco)
    summarize_split("Test  15%", test_coco)

    print("\nDone.")

if __name__ == "__main__":
    main()
