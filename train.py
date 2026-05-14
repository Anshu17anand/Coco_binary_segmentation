import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.coco_dataset import CocoBinarySegmentation
from models.unet import UNet
import segmentation_models_pytorch as smp

# -------- config --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 50
CATEGORY_ID = 18  # dog
# ------------------------

def load_ids(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f]


def main():
    # Dataset & DataLoader
    train_ids = load_ids("datasets/splits/train_ids.txt")
    dataset = CocoBinarySegmentation(
        images_dir="data/images/subset",
        ann_file="data/annotations/instances_train2017.json",
        category_id=CATEGORY_ID,
        image_ids=train_ids
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Model
    model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

    # Loss & Optimizer
    from losses import bce_dice_loss
    criterion = bce_dice_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Training loop
    model.train() #enables training mode
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)  #compares predictions vs ground truth, produces a single scalar

            # Backward pass
            optimizer.zero_grad()
            loss.backward()  # computes gradients for all the weights
            optimizer.step()  # nudges weights in the direction that reduces loss

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "unet_dog_segmentation.pth")
    print("Model saved to unet_dog_segmentation.pth")

    

if __name__ == "__main__":
    main()

