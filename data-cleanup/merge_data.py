import json
from pathlib import Path

train_file = "data/train_annotations.coco.json"
valid_file = "data/valid_annotations.coco.json"
test_file  = "data/test_annotations.coco.json"

merged_file = "data/merged_annotations.coco.json"

def merge_coco(files, out_file):
    merged = None
    img_id_offset = 0
    ann_id_offset = 0

    all_images = []
    all_annotations = []

    for f in files:
        with open(f, "r") as infile:
            data = json.load(infile)

        
        if merged is None:
            merged = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "categories": data.get("categories", []),
                "images": [],
                "annotations": []
            }

        img_id_map = {}
        for img in data["images"]:
            new_id = img["id"] + img_id_offset
            img_id_map[img["id"]] = new_id
            img["id"] = new_id
            all_images.append(img)

        for ann in data["annotations"]:
            ann["id"] = ann["id"] + ann_id_offset
            ann["image_id"] = img_id_map[ann["image_id"]]
            all_annotations.append(ann)

        img_id_offset = max([img["id"] for img in all_images]) + 1
        ann_id_offset = max([ann["id"] for ann in all_annotations]) + 1

    merged["images"] = all_images
    merged["annotations"] = all_annotations

    with open(out_file, "w") as outfile:
        json.dump(merged, outfile, indent=2)

    print(f"Merged dataset saved to {out_file}")
    print(f"Total images: {len(all_images)}, Total annotations: {len(all_annotations)}")

merge_coco([train_file, valid_file, test_file], merged_file)
