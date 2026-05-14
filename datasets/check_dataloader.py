from torch.utils.data import DataLoader
from coco_dataset import CocoBinarySegmentation

dataset = CocoBinarySegmentation(
    images_dir="data/images/subset",
    ann_file="data/annotations/instances_train2017.json",
    category_id=18
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

images, masks = next(iter(loader))

print("Batch images shape:", images.shape)
print("Batch masks shape:", masks.shape)
