"""손 궤적 특징 추출 (진위검출).

입력은 손인식 HandResult의 raw_trajectory (N, 2) float(EMA 전, 문제 이미지 픽셀)와 timestamps (N,) 초.
궤적 끝에서 손이 멈춘 구간(dwell, 손인식 멈춤 판정과 같은 기준)과 그 앞의 이동 구간으로 나눠서 본다.

특징
- speed_cv:     이동 구간 속도의 변동계수(표준편차/평균). 등속 봇은 0에 가깝다
- jitter_rms:   멈춤 구간의 떨림 크기(px). 원본 - 가우시안 스무딩(03강 가우시안 필터 연장)
                = 고주파 성분만 남긴 것의 RMS. 완전히 멈춘 봇은 0
- jitter_lag1:  그 고주파 성분의 lag-1 자기상관. 백색 노이즈 봇은 음수 쪽, 실제 떨림은 이웃 프레임끼리 이어져 있어 더 큼
- straightness: 이동 구간 직선거리 / 실제 경로 길이. 직선 봇은 1
- turn_std:     이동 구간 진행 방향 변화량(라디안)의 표준편차. 곡률이 일정하거나 0이면 작다
- max_speed:    중앙값 필터(3점) 뒤의 최대 속도(px/s). 순간이동 봇은 매우 크다.
                한 프레임만 튀는 인식 오류는 중앙값 필터로 걸러진다
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

STOP_SECONDS = 1.0      # 손인식 stop_seconds와 같게
STOP_PIXELS = 15.0      # 손인식 stop_pixels와 같게
JITTER_SIGMA = 2.0      # 고주파 분리용 가우시안 sigma (프레임 단위)
MIN_MOVE_SPEED = 30.0   # 이 속도(px/s) 미만은 방향이 의미 없어 turn_std에서 뺀다


@dataclass
class TrajectoryFeatures:
    speed_cv: float
    jitter_rms: float
    jitter_lag1: float
    straightness: float
    turn_std: float
    max_speed: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------- 기본 연산

def speeds(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    """프레임 간 속도 (N-1,) px/s."""
    dt = np.clip(np.diff(times), 1e-6, None)
    return np.linalg.norm(np.diff(points, axis=0), axis=1) / dt


def gaussian_smooth(points: np.ndarray, sigma: float = JITTER_SIGMA) -> np.ndarray:
    """궤적을 시간축으로 가우시안 스무딩 (저주파 성분). 양 끝은 복제 패딩."""
    ksize = int(2 * round(3 * sigma) + 1)
    kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
    # 궤적을 (N, 1) 2채널 영상으로 보고 세로 방향으로만 흐림
    img = points.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.sepFilter2D(img, -1, np.ones((1, 1), np.float32), kernel,
                          borderType=cv2.BORDER_REPLICATE)
    return out.reshape(-1, 2).astype(np.float64)


def high_pass(points: np.ndarray, sigma: float = JITTER_SIGMA) -> np.ndarray:
    """원본 - 가우시안 스무딩 = 고주파(떨림) 성분."""
    return points - gaussian_smooth(points, sigma)


def lag1_autocorr(x: np.ndarray) -> float:
    """1차원 신호의 lag-1 자기상관. 분산이 0이면 0."""
    x = x - x.mean()
    denom = float((x * x).sum())
    return 0.0 if denom < 1e-12 else float((x[:-1] * x[1:]).sum() / denom)


def split_dwell(points: np.ndarray, times: np.ndarray, stop_seconds: float = STOP_SECONDS,
                stop_pixels: float = STOP_PIXELS) -> np.ndarray:
    """멈춤 구간 마스크. 끝에서부터 거슬러 올라가며 이동 범위가 stop_pixels 미만인 동안 True.
    손인식의 멈춤 판정과 같은 기준이라 최소 stop_seconds는 포함된다."""
    dwell = split_last(times, stop_seconds)
    start = int(np.argmax(dwell))
    while start > 0:
        span = np.ptp(points[start - 1:], axis=0).max()
        if span >= stop_pixels:
            break
        start -= 1
    dwell[start:] = True
    return dwell


def split_last(times: np.ndarray, seconds: float) -> np.ndarray:
    """마지막 seconds 동안 True."""
    return times >= times[-1] - seconds - 1e-9


# ---------------------------------------------------------------- 특징

def extract(points: np.ndarray, times: np.ndarray) -> TrajectoryFeatures:
    points = np.asarray(points, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    dwell = split_dwell(points, times)
    move = ~dwell
    move[np.argmax(dwell)] = True       # 이동 구간 끝점 = 멈춤 시작점 (경계 포함)

    # 속도 변동 (이동 구간)
    mp, mt = points[move], times[move]
    v = speeds(mp, mt) if len(mp) >= 2 else np.zeros(1)
    speed_cv = float(v.std() / v.mean()) if v.mean() > 1e-9 else 0.0

    # 떨림 (마지막 stop_seconds 구간 안에서만 고주파 분리. 이동→정지 꺾임이 섞이지 않게)
    hp = high_pass(points[split_last(times, STOP_SECONDS)])
    jitter_rms = float(np.sqrt((hp ** 2).sum(axis=1).mean()))
    jitter_lag1 = float(np.mean([lag1_autocorr(hp[:, 0]), lag1_autocorr(hp[:, 1])]))

    # 곧음·방향 변화 (이동 구간)
    seg = np.diff(mp, axis=0)
    path_len = float(np.linalg.norm(seg, axis=1).sum())
    chord = float(np.linalg.norm(mp[-1] - mp[0]))
    straightness = chord / path_len if path_len > 1e-9 else 1.0
    moving = v >= MIN_MOVE_SPEED
    if moving.sum() >= 3:
        ang = np.unwrap(np.arctan2(seg[moving, 1], seg[moving, 0]))
        turn_std = float(np.diff(ang).std())
    else:
        turn_std = 0.0

    # 최대 속도 (전체, 3점 중앙값 필터로 한 프레임 튀는 값 제거)
    v_all = speeds(points, times)
    if len(v_all) >= 3:
        v_all = np.median(np.stack([v_all[:-2], v_all[1:-1], v_all[2:]]), axis=0)
    max_speed = float(v_all.max())

    return TrajectoryFeatures(speed_cv, jitter_rms, jitter_lag1, straightness, turn_std, max_speed)


if __name__ == "__main__":
    # 봇 목데이터 vs 사람 녹화의 특징 분포 비교
    import glob
    from pathlib import Path

    from cv.bot_detection.mock import load_csv
    from cv.bot_detection.record import load_recording

    data_dir = Path(__file__).resolve().parents[1] / "test" / "data"
    groups: dict[str, list[TrajectoryFeatures]] = {}
    for f in sorted(glob.glob(str(data_dir / "bot" / "*.csv"))):
        m = load_csv(f)
        groups.setdefault(m.kind, []).append(extract(m.raw_trajectory, m.timestamps))
    for f in sorted(glob.glob(str(data_dir / "human" / "*.npz"))):
        d = load_recording(f)
        groups.setdefault("human", []).append(extract(d["raw_trajectory"], d["timestamps"]))

    names = list(TrajectoryFeatures.__dataclass_fields__)
    print(f"{'group':13s}" + "".join(f"{n:>22s}" for n in names))
    for g, feats in groups.items():
        cells = []
        for n in names:
            vals = np.array([getattr(x, n) for x in feats])
            cells.append(f"{vals.min():9.2f} ~ {vals.max():9.2f}")
        print(f"{g:13s}" + "".join(f"{c:>22s}" for c in cells) + f"   (n={len(feats)})")
