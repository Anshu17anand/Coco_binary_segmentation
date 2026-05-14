import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CocoBinarySegmentation(Dataset):
    def __init__(self, images_dir, ann_file, category_id, image_ids=None):
        self.coco = COCO(ann_file)
        self.images_dir = images_dir
        self.category_id = category_id

        if image_ids is None:
            all_image_ids = self.coco.getImgIds(catIds=[category_id])
        else:
            all_image_ids = image_ids

        # keep only images that actually exist on disk
        self.image_ids = []
        for img_id in all_image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_dir, img_info["file_name"])
            if os.path.exists(img_path):
                self.image_ids.append(img_id)

        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2(),
        ])
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[self.category_id])
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            ann_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, ann_mask)

        ys, xs = np.where(mask > 0)

        if len(xs) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # add padding
            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(width, x_max + pad)
            y_max = min(height, y_max + pad)

            image = image[y_min:y_max, x_min:x_max]
            mask = mask[y_min:y_max, x_min:x_max]

        # resize image and mask to fixed size
        TARGET_SIZE = (256, 256)
        aug = self.transform(image=image, mask=mask)
        image = aug["image"]
        mask = aug["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)

        image = image.float() / 255.0

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.unsqueeze(0).float()
        mask = (mask > 0).float()
        return image, mask
