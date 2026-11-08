"""CNN 실행 (앱용, PyTorch 없이 OpenCV만): HandResult → '가짜일 확률'.

모델은 Kaggle에서 PyTorch로 학습(train.py) → ONNX로 변환(model.onnx) → 여기서 cv2.dnn으로 실행.
PyTorch로 돌린 값과 같음을 확인함 (차이 < 1e-8).

입력 변환 (Jester 학습 조건에 맞춤)
- 이동 구간(멈춤 전)만 씀 — 규칙 ②'와 같은 구간
- 초당 12장으로 다시 뽑기: 1/12초 간격 시각마다 가장 가까운 crop (반 칸 0.042초 안에 없으면 끊김)
- 쓸 수 없는 crop 제외: 크기 깨짐·전부 검정·직전과 픽셀까지 같은 중복 프레임 (frame_check와 같은 기준)
- 끊김 없이 이어진 10장 = 묶음 1개 (5장씩 밀며 여러 개), crop → 64×64 흑백, (x/255 − 0.5) / 0.25
- 녹화 점수 = 묶음들의 가짜 확률 중앙값. 묶음이 없으면 None (CNN 판단 생략)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.bot_detection.frame_check import frame_valid
from backend.bot_detection.trajectory import split_dwell

HERE = Path(__file__).resolve().parent
DEFAULT_ONNX = HERE / "model.onnx"
SEQ_LEN, SIZE = 10, 64
FPS = 12.0              # Jester 프레임 속도
STRIDE = 5              # 묶음 시작 간격 (장)
NORM = (0.5, 0.25)      # train.py 정규화


def to_input(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def windows(frames, frame_times, raw_trajectory, timestamps) -> np.ndarray:
    """녹화 → CNN 입력 묶음들 (M, 10, 64, 64) uint8. 없으면 (0, 10, 64, 64)."""
    n = len(frames) if frames is not None else 0
    empty = np.empty((0, SEQ_LEN, SIZE, SIZE), np.uint8)
    if n < SEQ_LEN:
        return empty
    t = np.asarray(frame_times if frame_times is not None and len(frame_times) == n else timestamps, np.float64)
    if len(t) != n:
        return empty
    ok = frame_valid(frames)
    for k in range(1, n):                                   # 중복 프레임 (카메라가 같은 프레임을 두 번 줌)
        if ok[k] and ok[k - 1] and np.array_equal(frames[k], frames[k - 1]):
            ok[k] = False
    dwell = split_dwell(np.asarray(raw_trajectory, np.float64), np.asarray(timestamps, np.float64))
    if len(dwell) == n:
        ok &= ~dwell
    idx = np.flatnonzero(ok)
    if len(idx) < SEQ_LEN:
        return empty

    grid = np.arange(t[idx[0]], t[idx[-1]] + 1e-9, 1.0 / FPS)   # 1/12초 격자
    picks: list[int | None] = []
    for g in grid:
        j = idx[np.argmin(np.abs(t[idx] - g))]
        good = abs(t[j] - g) <= 0.5 / FPS and (not picks or picks[-1] is None or j > picks[-1])
        picks.append(int(j) if good else None)

    out, run = [], []
    for p in picks + [None]:
        if p is None:
            for s in range(0, len(run) - SEQ_LEN + 1, STRIDE):
                out.append(np.stack([to_input(frames[q]) for q in run[s:s + SEQ_LEN]]))
            run = []
        else:
            run.append(p)
    return np.stack(out) if out else empty


class CnnScorer:
    """model.onnx를 cv2.dnn으로 읽어 묶음별 가짜 확률을 낸다."""

    def __init__(self, path: str | Path = DEFAULT_ONNX):
        # 경로를 넘기면 한글 경로(바탕 화면 등)에서 못 읽어서 바이트로 넘긴다
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(Path(path).read_bytes(), np.uint8))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """묶음들 (M,10,64,64) uint8 → 가짜일 확률 (M,)."""
        if len(x) == 0:
            return np.empty(0)
        out = []
        for s in range(0, len(x), 64):
            blob = (x[s:s + 64].astype(np.float32) / 255.0 - NORM[0]) / NORM[1]
            self.net.setInput(np.ascontiguousarray(blob))
            logits = self.net.forward()
            e = np.exp(logits - logits.max(1, keepdims=True))
            out.append((e / e.sum(1, keepdims=True))[:, 1])
        return np.concatenate(out)

    def score(self, frames, frame_times, raw_trajectory, timestamps) -> tuple[float | None, int]:
        """녹화 하나의 (가짜 확률 중앙값, 묶음 수). 묶음이 없으면 (None, 0)."""
        p = self.predict(windows(frames, frame_times, raw_trajectory, timestamps))
        return (float(np.median(p)) if len(p) else None), len(p)


_SCORER: CnnScorer | None = None
_LOADED = False


def get_scorer() -> CnnScorer | None:
    """model.onnx가 있으면 한 번만 읽어 재사용. 없으면 None (CNN 검사 생략)."""
    global _SCORER, _LOADED
    if not _LOADED:
        _LOADED = True
        if DEFAULT_ONNX.exists():
            _SCORER = CnnScorer(DEFAULT_ONNX)
    return _SCORER
