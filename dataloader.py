import os, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from utils import random_rotation, random_scaling, random_brightness, random_contrast, add_gaussian_noise
import random

class CrackDataset(Dataset):
    def __init__(self, json_path, img_dir, input_size=(512,512), augment=False):
        self.coco = COCO(json_path)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.input_size = input_size
        self.augment = augment
        self.augmentations = [
            random_rotation,
            random_scaling,
            random_brightness,
            random_contrast,
            add_gaussian_noise
        ]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info['file_name'])
        img = cv2.imread(path)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        if self.augment and random.random() > 0.5:
            aug_fn = random.choice(self.augmentations)
            img, mask = aug_fn(img, mask)

        img = cv2.resize(img, self.input_size)
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
        
        mask = (mask > 0).astype(np.float32)

        img = img.astype(np.float32)/255.0
        img = torch.from_numpy(img.transpose(2,0,1))
        mask = torch.from_numpy(mask).long()

        return img, mask

    def __len__(self):
        return len(self.ids)

def get_loader(json_path, img_dir, batch_size=4, shuffle=True, augment=False):
    dataset = CrackDataset(json_path, img_dir, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)