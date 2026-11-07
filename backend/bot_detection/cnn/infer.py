"""CNN 평가 (개발용, PyTorch 필요): 우리 녹화·합성 봇에 학습된 model.pt를 돌려 성적을 본다.
앱에서는 PyTorch 없이 runtime.py(ONNX + cv2.dnn)를 쓴다. 입력 변환(windows)은 둘이 같은 코드.

실행
    python -m backend.bot_detection.cnn.infer
    python -m backend.bot_detection.cnn.infer --jester ../handCaptch_cnn_results   # Jester 검증 세트 재현도 같이
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from backend.bot_detection.cnn.runtime import windows      # 입력 변환은 앱(runtime)과 같은 코드
from backend.bot_detection.cnn.train import SEQ_LEN, HandNet

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model.pt"


def load_model(path: str | Path = DEFAULT_MODEL, device: str = "cpu") -> tuple[HandNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HandNet(in_ch=ckpt.get("in_ch", SEQ_LEN))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt


@torch.no_grad()
def predict(model: HandNet, x: np.ndarray, norm=(0.5, 0.25), device: str = "cpu") -> np.ndarray:
    """묶음들 (M,10,64,64) uint8 → 가짜일 확률 (M,)."""
    if len(x) == 0:
        return np.empty(0)
    xt = (torch.from_numpy(x.astype(np.float32)) / 255.0 - norm[0]) / norm[1]
    out = []
    for s in range(0, len(xt), 256):
        out.append(torch.softmax(model(xt[s:s + 256].to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def score(model: HandNet, frames, frame_times, raw_trajectory, timestamps, norm=(0.5, 0.25)) -> tuple[float | None, int]:
    """녹화 하나의 (가짜 확률 중앙값, 묶음 수). 묶음이 없으면 (None, 0)."""
    p = predict(model, windows(frames, frame_times, raw_trajectory, timestamps), norm)
    return (float(np.median(p)) if len(p) else None), len(p)


# ---------------------------------------------------------------- 평가

def _jester_check(model, ckpt, root: Path) -> None:
    """Kaggle 검증 결과가 로컬에서도 재현되는지 (입력 변환·모델 로드 확인용)."""
    ps, ys = [], []
    for prefix, y in (("real_", 0), ("fake_", 1)):
        for f in sorted(root.rglob(f"{prefix}*.npz")):
            if "test_" in f.parent.name:
                continue
            d = np.load(f)
            m = d["split"] == 1
            ps.append(predict(model, d["x"][m], ckpt.get("norm", (0.5, 0.25))))
            ys.append(np.full(int(m.sum()), y))
    p, y = np.concatenate(ps), np.concatenate(ys)
    print(f"[Jester 검증 재현] {len(y)}개, 정확도 {((p > 0.5) == (y == 1)).mean():.3f}, "
          f"사람 오판 {(p[y == 0] > 0.5).mean():.3f}, 가짜 검출 {(p[y == 1] > 0.5).mean():.3f}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data", default=str(HERE.parent / "test" / "data"))
    ap.add_argument("--jester", default=None, help="Kaggle 결과 폴더 (jester_real/fake npz) — 검증 재현용")
    args = ap.parse_args()

    model, ckpt = load_model(args.model)
    norm = ckpt.get("norm", (0.5, 0.25))
    thr_strict = float(ckpt.get("threshold_1pct", 0.976))
    if args.jester:
        _jester_check(model, ckpt, Path(args.jester))

    data = Path(args.data)
    groups = {"human": sorted(glob.glob(str(data / "human" / "*.npz")))}
    for lvl in ("S1", "S2", "S3", "S4"):
        groups[lvl] = sorted(glob.glob(str(data / "synth" / lvl / "*.npz")))

    print(f"녹화 점수 = 10장 묶음들의 가짜 확률 중앙값. 문턱 0.5 / {thr_strict:.3f}(Jester에서 사람 오판 ≤1%)")
    print("그룹   | 녹화 | 묶음 없음 | 점수 최소 / 중앙 / 최대 | >0.5 (봇 판정) | >%.3f (봇 판정)" % thr_strict)
    detail = {}
    for g, files in groups.items():
        scores, none = [], 0
        for f in files:
            d = np.load(f)
            s, m = score(model, d["frames"], d["frame_timestamps"], d["raw_trajectory"], d["timestamps"], norm)
            detail[Path(f).stem] = (s, m)
            if s is None:
                none += 1
            else:
                scores.append(s)
        a = np.array(scores)
        if len(a):
            print(f"{g:6} | {len(files):4} | {none:9} | {a.min():.2f} / {np.median(a):.2f} / {a.max():.2f}"
                  f"        | {int((a > 0.5).sum()):3} / {len(a):<3}       | {int((a > thr_strict).sum()):3} / {len(a)}")
        else:
            print(f"{g:6} | {len(files):4} | {none:9} | -")
    print("\n사람 중 점수 높은 순 (봇으로 오해받기 쉬운 녹화):")
    hs = sorted(((v[0], k, v[1]) for k, v in detail.items() if k.startswith("human") and v[0] is not None), reverse=True)
    for s, k, m in hs[:8]:
        print(f"  {s:.3f}  {k}  (묶음 {m}개)")


if __name__ == "__main__":
    main()
