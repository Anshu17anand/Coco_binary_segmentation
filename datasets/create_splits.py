import random
from pycocotools.coco import COCO
ANN_FILE = "data/annotations/instances_train2017.json"
CATEGORY_ID = 18  # dog
OUTPUT_DIR = "datasets/splits"
MAX_IMAGES = 250
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42

def main():
    coco = COCO(ANN_FILE)
    import os
    image_files = os.listdir("data/images/subset")
    img_ids = [int(f.split(".")[0]) for f in image_files if f.endswith(".jpg")]
    print(f"Total images with dogs: {len(img_ids)}")

    random.seed(SEED)
    random.shuffle(img_ids)

# cap total number of images
    img_ids = img_ids[:MAX_IMAGES]

    n = len(img_ids)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)

    train_ids = img_ids[:n_train]
    val_ids = img_ids[n_train:n_train + n_val]
    test_ids = img_ids[n_train + n_val:]

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save(name, ids):
        with open(f"{OUTPUT_DIR}/{name}.txt", "w") as f:
            for i in ids:
                f.write(f"{i}\n")

    save("train_ids", train_ids)
    save("val_ids", val_ids)
    save("test_ids", test_ids)

    print("Split sizes:")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

if __name__ == "__main__":
    main()
