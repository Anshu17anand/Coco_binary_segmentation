import torch
import matplotlib.pyplot as plt
import random
from datasets.coco_dataset import CocoBinarySegmentation
import segmentation_models_pytorch as smp
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORY_ID = 18

# Load dataset
dataset = CocoBinarySegmentation(
    images_dir="data/images/subset",
    ann_file="data/annotations/instances_train2017.json",
    category_id=CATEGORY_ID,
)

# Load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # IMPORTANT
    in_channels=3,
    classes=1,
).to(DEVICE)

model.load_state_dict(torch.load("unet_dog_segmentation.pth", map_location=DEVICE))
model.eval()


idx = random.randint(0, len(dataset) - 1)
image, gt_mask = dataset[idx]
print(f"Visualizing sample index: {idx}")


with torch.no_grad():
    image_batch = image.unsqueeze(0).to(DEVICE)
    logits = model(image_batch)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > 0.4).float()

    pred_np = pred_mask.cpu().numpy().astype(np.uint8)[0][0]
    kernel = np.ones((3,3), np.uint8)
    pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)

    pred = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)

# Convert to numpy for plotting
image_np = image.permute(1, 2, 0).cpu().numpy()
gt_mask_np = gt_mask.squeeze().cpu().numpy()
pred_mask_np = pred.squeeze().cpu().numpy()

# Plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(gt_mask_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred_mask_np, cmap="gray")
plt.axis("off")

plt.savefig("prediction_visualization.png")
print("Saved prediction_visualization.png")
