import torch
from torch.utils.data import DataLoader
from datasets.coco_dataset import CocoBinarySegmentation
from models.unet import UNet
from losses import dice_loss
import segmentation_models_pytorch as smp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORY_ID = 18

def load_ids(path):
    with open(path) as f:
        return [int(x.strip()) for x in f.readlines()]
    
def iou_score(preds, targets, eps=1e-6):
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def evaluate(split_name, ids_path):
    dataset = CocoBinarySegmentation(
        images_dir="data/images/subset",
        ann_file="data/annotations/instances_train2017.json",
        category_id=CATEGORY_ID,
        image_ids=load_ids(ids_path),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # IMPORTANT
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load("unet_dog_segmentation.pth", map_location=DEVICE))
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.4).float()

            # Dice (same as before)
            loss = dice_loss(logits, masks)
            total_dice += (1 - loss).item()

            # IoU (NEW)
            iou = iou_score(preds, masks)
            total_iou += iou

            count += 1

    avg_dice = total_dice / count
    avg_iou = total_iou / count
    print(f"{split_name} Dice: {avg_dice:.4f}")
    print(f"{split_name} IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    evaluate("Val", "datasets/splits/val_ids.txt")
    evaluate("Test", "datasets/splits/test_ids.txt")
