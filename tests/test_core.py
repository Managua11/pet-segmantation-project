import os

import cv2
import numpy as np
import torch

from src.dataset import PetDataset
from src.metrics import dice_score, iou_score
from src.model import build_model
from src.predict import preprocess

ROOT = "data/oxford-iiit-pet"
NAME = "Abyssinian_1"
IMAGE = os.path.join(ROOT, "images", f"{NAME}.jpg")
MASK = os.path.join(ROOT, "annotations", "trimaps", f"{NAME}.png")


def test_dataset_item():
    image, mask = PetDataset([IMAGE], [MASK], img_size=256)[0]
    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_model_output_shape():
    model = build_model(weights=None).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 256, 256)).shape == (2, 1, 256, 256)


def test_metrics():
    targets = torch.zeros(2, 1, 8, 8)
    targets[:, :, :4] = 1
    perfect = targets * 20 - 10  # დადებითი logit იქ, სადაც 1-ია
    assert abs(iou_score(perfect, targets) - 1) < 1e-4
    assert abs(dice_score(perfect, targets) - 1) < 1e-4
    assert iou_score(-perfect, targets) < 1e-4
    assert dice_score(-perfect, targets) < 1e-4


def test_preprocess_matches_dataset():
    image = cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB)
    expected = PetDataset([IMAGE], [MASK], img_size=256, augment=False)[0][0]
    assert torch.allclose(preprocess(image, 256), expected, atol=1e-5)


def test_onnx_matches_pytorch(tmp_path):
    import onnxruntime as ort  # torch უკვე დაიმპორტებულია
    from src.optimize import export_onnx

    model = build_model(weights=None).eval()
    path = str(tmp_path / "model.onnx")
    export_onnx(model, path, 256)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        expected = model(x).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    assert np.allclose(out, expected, atol=1e-4)
