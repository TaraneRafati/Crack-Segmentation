import json
import random
from pathlib import Path
from collections import Counter, defaultdict

MERGED_FILE = "data/merged_annotations.coco.json"  
OUT_DIR = Path(".")                          
TRAIN_OUT = OUT_DIR / "data/train_.coco.json"
VAL_OUT   = OUT_DIR / "data/valid_.coco.json"
TEST_OUT  = OUT_DIR / "data/test_.coco.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
REINDEX_IDS_PER_SPLIT = False   

def subset_coco(base, images, annotations, reindex=False):
    """Create a COCO dict for a subset. Optionally reindex IDs starting from 1."""
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
        img_copy = dict(img)
        img_id_map[img["id"]] = new_id
        img_copy["id"] = new_id
        new_images.append(img_copy)

    new_annotations = []
    for new_ann_id, ann in enumerate(annotations, start=1):
        ann_copy = dict(ann)
        ann_copy["id"] = new_ann_id
        ann_copy["image_id"] = img_id_map[ann["image_id"]]
        new_annotations.append(ann_copy)

    coco["images"] = new_images
    coco["annotations"] = new_annotations
    return coco

def summarize_split(name, images, annotations, categories):
    img_count = len(images)
    ann_count = len(annotations)
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_counts = Counter([a["category_id"] for a in annotations])
    cat_str = ", ".join(f"{cat_id_to_name.get(k, k)}={v}" for k, v in sorted(cat_counts.items()))
    print(f"[{name}] images={img_count:4d} | anns={ann_count:5d} | per-category: {cat_str}")

def main():
    random.seed(SEED)

    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    images_shuffled = images[:] 
    random.shuffle(images_shuffled)

    
    N = len(images_shuffled)
    n_train = int(N * TRAIN_RATIO)
    n_val = int(N * VAL_RATIO)
    n_test = N - n_train - n_val  

    train_imgs = images_shuffled[:n_train]
    val_imgs   = images_shuffled[n_train:n_train + n_val]
    test_imgs  = images_shuffled[n_train + n_val:]

    train_ids = {img["id"] for img in train_imgs}
    val_ids   = {img["id"] for img in val_imgs}
    test_ids  = {img["id"] for img in test_imgs}

    train_anns = [a for a in annotations if a["image_id"] in train_ids]
    val_anns   = [a for a in annotations if a["image_id"] in val_ids]
    test_anns  = [a for a in annotations if a["image_id"] in test_ids]

    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids), \
        "Image ID overlap detected between splits"

    train_coco = subset_coco(data, train_imgs, train_anns, reindex=REINDEX_IDS_PER_SPLIT)
    val_coco   = subset_coco(data, val_imgs, val_anns, reindex=REINDEX_IDS_PER_SPLIT)
    test_coco  = subset_coco(data, test_imgs, test_anns, reindex=REINDEX_IDS_PER_SPLIT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2)
    with open(VAL_OUT, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, indent=2)
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        json.dump(test_coco, f, indent=2)

    print("Done. New splits written:")
    print(f" - {TRAIN_OUT}")
    print(f" - {VAL_OUT}")
    print(f" - {TEST_OUT}")
    summarize_split("Train", train_coco["images"], train_coco["annotations"], categories)
    summarize_split("Valid", val_coco["images"],   val_coco["annotations"],   categories)
    summarize_split("Test ", test_coco["images"],  test_coco["annotations"],  categories)

if __name__ == "__main__":
    main()
