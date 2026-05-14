import matplotlib.pyplot as plt
from coco_dataset import CocoBinarySegmentation

dataset = CocoBinarySegmentation(
    images_dir="data/images/subset",
    ann_file="data/annotations/instances_train2017.json",
    category_id=18  # dog
)

image, mask = dataset[0]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Dog Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.savefig("sample_visualization.png")
print("Saved visualization to sample_visualization.png")

print("Mask foreground pixels:", mask.sum())
print("Total pixels:", mask.shape[0] * mask.shape[1])

