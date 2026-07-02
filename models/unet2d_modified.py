import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_channels, out_channels, use_batch_norm=True):
    layers = []

    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class _ModifiedUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_classes, use_batch_norm=True):
        super().__init__()

        # The paper's modified 2D U-Net keeps the transposed-convolution
        # upsampling path narrow by setting its output channels to num_classes.
        upsample_layers = [
            nn.ConvTranspose2d(in_channels, num_classes, kernel_size=2, stride=2),
        ]
        if use_batch_norm:
            upsample_layers.append(nn.BatchNorm2d(num_classes))
        upsample_layers.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*upsample_layers)
        self.conv = _conv_block(num_classes + skip_channels, out_channels, use_batch_norm)

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ModifiedUNet2D(nn.Module):
    """2D U-Net variant with narrow transposed-convolution upsampling layers."""

    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        base_channels=64,
        use_batch_norm=True,
    ):
        super().__init__()

        self.encoder1 = _conv_block(in_channels, base_channels, use_batch_norm)
        self.encoder2 = _conv_block(base_channels, base_channels * 2, use_batch_norm)
        self.encoder3 = _conv_block(base_channels * 2, base_channels * 4, use_batch_norm)
        self.encoder4 = _conv_block(base_channels * 4, base_channels * 8, use_batch_norm)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _conv_block(base_channels * 8, base_channels * 16, use_batch_norm)

        self.decoder4 = _ModifiedUpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
            num_classes,
            use_batch_norm,
        )
        self.decoder3 = _ModifiedUpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            num_classes,
            use_batch_norm,
        )
        self.decoder2 = _ModifiedUpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            num_classes,
            use_batch_norm,
        )
        self.decoder1 = _ModifiedUpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            num_classes,
            use_batch_norm,
        )

        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))

        x = self.bottleneck(self.pool(skip4))

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.classifier(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
