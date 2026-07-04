import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        num_classes = logits.shape[1]
        probabilities = torch.softmax(logits, dim=1)
        targets = F.one_hot(labels, num_classes=num_classes)
        targets = targets.movedim(-1, 1).to(dtype=probabilities.dtype)

        reduce_dims = tuple(dim for dim in range(probabilities.ndim) if dim not in (0, 1))
        intersection = (probabilities * targets).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return 1 - dice[:, 1:].mean()


def build_loss(loss_name, num_classes, device, background_weight=0.1, foreground_weight=0.3):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if loss_name == "weighted_cross_entropy":
        weights = torch.full((num_classes,), foreground_weight, dtype=torch.float32, device=device)
        weights[0] = background_weight
        return nn.CrossEntropyLoss(weight=weights)

    if loss_name == "dice":
        return ForegroundDiceLoss()

    raise ValueError(f"Unknown loss function: {loss_name}")


def add_loss_arguments(parser):
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "weighted_cross_entropy", "dice"],
        default="cross_entropy",
        help="Loss function to optimize. Options match the losses compared in Baumgartner et al.",
    )
    parser.add_argument(
        "--background-class-weight",
        type=float,
        default=0.1,
        help="Background weight for --loss weighted_cross_entropy, matching the paper default.",
    )
    parser.add_argument(
        "--foreground-class-weight",
        type=float,
        default=0.3,
        help="Foreground class weight for --loss weighted_cross_entropy, matching the paper default.",
    )
