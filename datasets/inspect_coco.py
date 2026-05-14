import json

ann_file = "data/annotations/instances_train2017.json"

with open(ann_file, "r") as f:
    coco = json.load(f)

for cat in coco["categories"]:
    if cat["name"] == "dog":
        print(cat)
