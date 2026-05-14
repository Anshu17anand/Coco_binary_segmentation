from coco_dataset import CocoBinarySegmentation

dataset = CocoBinarySegmentation(
    images_dir="data/images/subset",
    ann_file="data/annotations/instances_train2017.json",
    category_id=18
)

image, mask = dataset[0]

print("Image shape:", image.shape, image.dtype)
print("Mask shape:", mask.shape, mask.dtype)
print("Image min/max:", image.min().item(), image.max().item())
print("Mask unique values:", mask.unique())
