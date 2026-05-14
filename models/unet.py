import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()

        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for f in features:
            self.downs.append(DoubleConv(prev_channels, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_features = features[::-1]
        prev_channels = features[-1] * 2
        for f in rev_features:
            self.ups.append(
                nn.ConvTranspose2d(prev_channels, f, kernel_size=2, stride=2)
            )
            self.up_convs.append(DoubleConv(prev_channels, f))
            prev_channels = f

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for up, conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)

            # In case of odd sizes (shouldn’t happen with 256, but safe)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = conv(x)

        return self.final_conv(x)
