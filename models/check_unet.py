import torch
from unet import UNet

model = UNet()
x = torch.randn(1, 3, 256, 256)
y = model(x)

print("Output shape:", y.shape)

# Parameter count
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters:", num_params)
