import cv2
import numpy as np
import torch

from src.dataset import MEAN, STD
from src.model import build_model


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["args"]["encoder"], weights=None)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["args"]["img_size"]


def preprocess(image_rgb, img_size):
    image = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image = (image / 255.0 - MEAN) / STD
    return torch.from_numpy(image).permute(2, 0, 1).float()


def predict(model, image_rgb, img_size, threshold=0.5):
    h, w = image_rgb.shape[:2]
    with torch.no_grad():
        logits = model(preprocess(image_rgb, img_size).unsqueeze(0))
    prob = torch.sigmoid(logits)[0, 0].numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8)


def overlay(image_rgb, mask, alpha=0.5):
    out = image_rgb.astype(np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    pet = mask == 1
    out[pet] = (1 - alpha) * out[pet] + alpha * green
    return out.astype(np.uint8)
