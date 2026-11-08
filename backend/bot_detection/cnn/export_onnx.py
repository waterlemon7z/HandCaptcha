"""학습된 model.pt(PyTorch) → model.onnx 변환. 앱은 model.onnx를 cv2.dnn으로 읽어서 PyTorch가 필요 없다.

실행 (PyTorch + onnx 필요, 개발할 때 한 번만)
    pip install torch onnx
    python -m backend.bot_detection.cnn.export_onnx
변환 후 같은 입력으로 PyTorch와 cv2.dnn 결과가 같은지 확인한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.bot_detection.cnn.infer import load_model

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "model.pt"))
    ap.add_argument("--out", default=str(HERE / "model.onnx"))
    args = ap.parse_args()

    model, ckpt = load_model(args.model)
    torch.onnx.export(model, torch.zeros(1, 10, 64, 64), args.out, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=13, dynamo=False)

    x = torch.randn(4, 10, 64, 64)
    with torch.no_grad():
        ref = torch.softmax(model(x), 1)[:, 1].numpy()
    net = cv2.dnn.readNetFromONNX(np.frombuffer(Path(args.out).read_bytes(), np.uint8))
    net.setInput(x.numpy())
    logits = net.forward()
    e = np.exp(logits - logits.max(1, keepdims=True))
    got = (e / e.sum(1, keepdims=True))[:, 1]
    print(f"저장: {args.out}  (PyTorch vs cv2.dnn 최대 차이 {np.abs(got - ref).max():.2e}, "
          f"문턱 threshold_1pct={ckpt.get('threshold_1pct')})")


if __name__ == "__main__":
    main()
