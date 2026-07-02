import torch
import torch.nn as nn


def _conv_block(in_channels, out_channels, num_convs, use_batch_norm=True):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    return nn.Sequential(*layers)


def _make_bilinear_weights(layer):
    """Initialize a transposed convolution to behave like bilinear upsampling."""
    kernel_size = layer.kernel_size[0]
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5

    og = torch.arange(kernel_size, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor).unsqueeze(0)
    kernel = filt.t() @ filt

    with torch.no_grad():
        layer.weight.zero_()
        for channel in range(min(layer.in_channels, layer.out_channels)):
            layer.weight[channel, channel] = kernel


class FCN8(nn.Module):
    """FCN-8 segmentation network with VGG-style encoder and skip connections."""

    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        classifier_channels=1024,
        use_batch_norm=True,
        dropout=0.5,
    ):
        super().__init__()

        self.block1 = _conv_block(in_channels, 64, num_convs=2, use_batch_norm=use_batch_norm)
        self.block2 = _conv_block(64, 128, num_convs=2, use_batch_norm=use_batch_norm)
        self.block3 = _conv_block(128, 256, num_convs=3, use_batch_norm=use_batch_norm)
        self.block4 = _conv_block(256, 512, num_convs=3, use_batch_norm=use_batch_norm)
        self.block5 = _conv_block(512, 512, num_convs=3, use_batch_norm=use_batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # These layers are the convolutional version of VGG's fully-connected head.
        self.classifier = nn.Sequential(
            nn.Conv2d(512, classifier_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(classifier_channels, classifier_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(classifier_channels, num_classes, kernel_size=1),
        )

        # Score layers convert intermediate feature maps into per-class logits.
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        # FCN-8 upsamples stride-32 logits to stride 16, then 8, then input size.
        self.upscore2 = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=16,
            stride=8,
            padding=4,
            bias=False,
        )

        self._initialize_weights()

    def forward(self, x):
        input_size = x.size()

        x = self.block1(x)
        pool1 = self.pool(x)

        x = self.block2(pool1)
        pool2 = self.pool(x)

        x = self.block3(pool2)
        pool3 = self.pool(x)

        x = self.block4(pool3)
        pool4 = self.pool(x)

        x = self.block5(pool4)
        pool5 = self.pool(x)

        score = self.classifier(pool5)

        score_pool4 = self.score_pool4(pool4)
        score = self.upscore2(score, output_size=score_pool4.size())
        score = score + score_pool4

        score_pool3 = self.score_pool3(pool3)
        score = self.upscore_pool4(score, output_size=score_pool3.size())
        score = score + score_pool3

        score = self.upscore8(score, output_size=input_size)
        return score

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        _make_bilinear_weights(self.upscore2)
        _make_bilinear_weights(self.upscore_pool4)
        _make_bilinear_weights(self.upscore8)
