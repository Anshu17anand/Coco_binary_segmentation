import os
import requests
from pycocotools.coco import COCO

ANN_FILE = "data/annotations/instances_train2017.json"
OUT_DIR = "data/images/subset"
CATEGORY_NAME = "dog"
NUM_IMAGES = 250

os.makedirs(OUT_DIR, exist_ok=True)

coco = COCO(ANN_FILE)

# get category id for dog
cat_id = coco.getCatIds(catNms=[CATEGORY_NAME])[0]

# get image ids that contain dogs
img_ids = coco.getImgIds(catIds=[cat_id])[:NUM_IMAGES]
imgs = coco.loadImgs(img_ids)

for img in imgs:
    url = img["coco_url"]
    file_path = os.path.join(OUT_DIR, img["file_name"])

    if os.path.exists(file_path):
        continue

    print(f"Downloading {img['file_name']}")
    r = requests.get(url, stream=True)
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
