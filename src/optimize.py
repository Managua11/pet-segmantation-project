import argparse
import os
import time

import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.utils.data import DataLoader

from src.dataset import get_splits
from src.metrics import iou_score
from src.predict import load_model

DATA_ROOT = "data/oxford-iiit-pet"
THREADS = 6


def export_onnx(model, path, img_size):
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)


def ort_runner(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return lambda x: torch.from_numpy(session.run(None, {"input": x.numpy()})[0])


def median_latency_ms(run, img_size):
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):
        run(x)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        run(x)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def val_iou(run, loader):
    total, n = 0.0, 0
    for images, masks in loader:
        total += iou_score(run(images), masks) * images.size(0)
        n += images.size(0)
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(THREADS)

    model, img_size = load_model(args.checkpoint)
    exp_dir = os.path.dirname(args.checkpoint)
    fp32_path = os.path.join(exp_dir, "model_fp32.onnx")
    int8_path = os.path.join(exp_dir, "model_int8.onnx")
    export_onnx(model, fp32_path, img_size)
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)

    def torch_run(x):
        with torch.no_grad():
            return model(x)

    _, val_ds = get_splits(DATA_ROOT, img_size=img_size)
    loader = DataLoader(val_ds, batch_size=16)
    variants = [("PyTorch FP32", args.checkpoint, torch_run),
                ("ONNX FP32", fp32_path, ort_runner(fp32_path)),
                ("ONNX INT8", int8_path, ort_runner(int8_path))]

    lines = ["| variant | size MB | median ms | val IoU |", "|---|---:|---:|---:|"]
    for name, path, run in variants:
        size_mb = os.path.getsize(path) / 1e6
        ms = median_latency_ms(run, img_size)
        iou = val_iou(run, loader)
        lines.append(f"| {name} | {size_mb:.1f} | {ms:.1f} | {iou:.4f} |")
        print(lines[-1])

    table = "\n".join(lines)
    os.makedirs("experiments/optimization", exist_ok=True)
    with open("experiments/optimization/results.md", "w", encoding="utf-8") as f:
        f.write(f"# Optimization results ({args.checkpoint})\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
