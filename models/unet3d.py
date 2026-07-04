import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_channels, out_channels, use_batch_norm=True):
    layers = []

    layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class _UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride, use_batch_norm=True):
        super().__init__()

        upsample_layers = [
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=stride, stride=stride),
        ]
        if use_batch_norm:
            upsample_layers.append(nn.BatchNorm3d(out_channels))
        upsample_layers.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*upsample_layers)
        self.conv = _conv_block(out_channels + skip_channels, out_channels, use_batch_norm)

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """Modified 3D U-Net for cardiac MR volumes.

    The architecture follows the paper's practical 3D variant by pooling along
    the through-plane/depth axis only once. Later stages pool only in-plane.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        base_channels=16,
        use_batch_norm=True,
    ):
        super().__init__()

        self.encoder1 = _conv_block(in_channels, base_channels, use_batch_norm)
        self.encoder2 = _conv_block(base_channels, base_channels * 2, use_batch_norm)
        self.encoder3 = _conv_block(base_channels * 2, base_channels * 4, use_batch_norm)
        self.encoder4 = _conv_block(base_channels * 4, base_channels * 8, use_batch_norm)

        self.pool_depth = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool_in_plane = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck = _conv_block(base_channels * 8, base_channels * 16, use_batch_norm)

        self.decoder4 = _UpBlock3D(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
            stride=(1, 2, 2),
            use_batch_norm=use_batch_norm,
        )
        self.decoder3 = _UpBlock3D(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            stride=(1, 2, 2),
            use_batch_norm=use_batch_norm,
        )
        self.decoder2 = _UpBlock3D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            stride=(1, 2, 2),
            use_batch_norm=use_batch_norm,
        )
        self.decoder1 = _UpBlock3D(
            base_channels * 2,
            base_channels,
            base_channels,
            stride=(2, 2, 2),
            use_batch_norm=use_batch_norm,
        )

        self.classifier = nn.Conv3d(base_channels, num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool_depth(skip1))
        skip3 = self.encoder3(self.pool_in_plane(skip2))
        skip4 = self.encoder4(self.pool_in_plane(skip3))

        x = self.bottleneck(self.pool_in_plane(skip4))

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.classifier(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
