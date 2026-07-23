import pytest
import torch

from models import FCN8, ModifiedUNet2D, UNet2D, UNet3D


@pytest.mark.parametrize(
    ("model", "shape"),
    [
        (FCN8(classifier_channels=16), (1, 1, 32, 32)),
        (UNet2D(base_channels=4), (1, 1, 32, 32)),
        (ModifiedUNet2D(base_channels=4), (1, 1, 32, 32)),
    ],
)
def test_2d_models_preserve_spatial_shape(model, shape):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(shape))
    assert output.shape == (shape[0], 4, shape[2], shape[3])


def test_3d_model_preserves_spatial_shape():
    model = UNet3D(base_channels=2)
    model.eval()
    shape = (1, 1, 4, 32, 32)
    with torch.no_grad():
        output = model(torch.randn(shape))
    assert output.shape == (shape[0], 4, shape[2], shape[3], shape[4])
