"""영상 검사 (진위검출 2단계): 이 crop들이 진짜 카메라로 찍은 진짜 손인가.

입력은 손인식 HandResult의 frames(검지 끝 중심 crop, 좌우 반전), raw_trajectory, frame_size.
crop 창의 위치(카메라 픽셀)는 raw_trajectory를 거꾸로 환산해서 구한다.

① 배경 시간 노이즈 (02-2 프레임 차이)
   멈춤 구간에서 crop 창 위치가 똑같은 연속 두 장의 absdiff.
   진짜 웹캠은 정지한 배경도 센서 노이즈로 픽셀이 흔들리지만, 합성 영상은 같은 사진이라 정확히 0이다.
   → zero_ratio: 밝기 차이가 정확히 0인 픽셀 비율 (합성일수록 큼)

② 손 변형 (04강 Harris 코너 + 03강 피라미드 Lucas-Kanade)
   이동 구간 연속 두 장에서 Harris 코너를 LK로 추적한다.
   crop 창이 손을 따라가므로 손 위의 점은 crop 안에서 거의 안 움직이고 배경은 창 이동의 반대(-Δ)로 밀린다.
   0에 더 가까이 움직인 점을 손으로 보고, 그 이동 벡터들이 서로 얼마나 다른지(퍼짐)를 잰다.
   손 사진을 평행이동만 하는 합성은 퍼짐이 0에 가깝고, 진짜 손은 손가락마다 조금씩 달리 움직인다.
   → deform: 손 점 이동 벡터의 중앙값으로부터의 거리 중앙값 (px)
   → affine_res: 손 점 이동을 회전+확대+이동 변환 하나(02-2 기하 변환)로 맞춘 뒤 남는 오차 중앙값 (px).
      손 사진을 돌리고 키운 합성(S4)은 변환 하나로 설명돼 ≈0, 손가락마다 따로 굽는 진짜 손은 남는다
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from cv.bot_detection.trajectory import split_dwell

PROBLEM_W, PROBLEM_H = 1280, 720
MIN_MOVE, MAX_MOVE = 3, 32      # ②에 쓸 crop 창 이동량 범위 (카메라 px). 너무 작으면 손/배경 구분 불가, 크면 안 겹침
MIN_HAND_POINTS = 4             # ②에서 손으로 분류된 점이 이보다 적으면 그 쌍은 건너뜀
FB_MAX_ERR = 0.5                # LK 앞→뒤 추적 오차가 이보다 크면 그 점은 버림 (px)


@dataclass
class FrameFeatures:
    zero_ratio: float           # ① 멈춤 구간 배경 차이가 0인 픽셀 비율 (nan = 쓸 쌍 없음)
    noise_pairs: int            # ①에 쓴 쌍 수
    deform: float               # ② 손 점 이동 벡터 퍼짐 (px, nan = 쓸 쌍 없음)
    deform_pairs: int           # ②에 쓴 쌍 수
    affine_res: float           # ② 손 점 이동을 회전·확대·이동 변환으로 맞춘 뒤 남는 오차 (px, nan = 쓸 쌍 없음)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------- crop 창 위치

def crop_centers(raw_trajectory: np.ndarray, frame_size) -> np.ndarray:
    """raw_trajectory(문제 이미지, 좌우 반전됨) → crop 창 중심 (카메라 정수 픽셀) (N, 2).
    손인식은 카메라 정규화 좌표 tx를 1-tx로 뒤집어 문제 이미지에 매핑하고, crop은 원래 카메라 좌표에서 자른다."""
    w, h = frame_size
    tx = 1.0 - raw_trajectory[:, 0] / (PROBLEM_W - 1)
    ty = raw_trajectory[:, 1] / (PROBLEM_H - 1)
    return np.rint(np.column_stack([tx * (w - 1), ty * (h - 1)])).astype(int)


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


# ---------------------------------------------------------------- ① 배경 시간 노이즈

def zero_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """두 장의 밝기 차이(absdiff)가 정확히 0인 픽셀 비율."""
    return float((cv2.absdiff(to_gray(a), to_gray(b)) == 0).mean())


def temporal_noise(frames, centers: np.ndarray, dwell: np.ndarray) -> tuple[float, int]:
    """멈춤 구간에서 창 위치가 같은 연속 쌍들의 zero_ratio 중앙값."""
    vals = [zero_ratio(frames[k], frames[k + 1])
            for k in range(len(frames) - 1)
            if dwell[k] and dwell[k + 1] and (centers[k] == centers[k + 1]).all()]
    return (float(np.median(vals)) if vals else float("nan")), len(vals)


# ---------------------------------------------------------------- ② 손 변형

def track_corners(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a의 Harris 코너를 b로 피라미드 LK 추적. 앞→뒤 추적이 어긋나는 점은 버린다. (시작점, 이동 벡터)."""
    ga, gb = to_gray(a), to_gray(b)
    pts = cv2.goodFeaturesToTrack(ga, maxCorners=60, qualityLevel=0.01, minDistance=4,
                                  useHarrisDetector=True, k=0.04)
    if pts is None:
        return np.empty((0, 2)), np.empty((0, 2))
    lk = dict(winSize=(15, 15), maxLevel=2)
    fwd, st1, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, fwd, None, **lk)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & \
         (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < FB_MAX_ERR)
    p = pts.reshape(-1, 2)[ok]
    return p, fwd.reshape(-1, 2)[ok] - p


def affine_residual(src: np.ndarray, dst: np.ndarray) -> float:
    """src→dst를 회전+확대+이동 변환 하나로 맞춘 뒤 남는 오차의 중앙값 (px).
    LMEDS(오차 제곱의 중앙값 최소화)로 맞춰, 손으로 잘못 분류된 배경 점 몇 개에 변환이 끌려가지 않게 한다."""
    m, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32),
                                       method=cv2.LMEDS)
    if m is None:
        return float("nan")
    pred = src @ m[:, :2].T + m[:, 2]
    return float(np.median(np.linalg.norm(pred - dst, axis=1)))


def pair_deform(a: np.ndarray, b: np.ndarray, window_move: np.ndarray) -> tuple[float, float] | None:
    """한 쌍의 (손 변형량, 변환 맞춤 오차). window_move = crop 창 이동 (crop 안 좌표, 좌우 반전 반영)."""
    pts, flow = track_corners(a, b)
    if len(flow) == 0:
        return None
    bg_flow = -window_move
    # 0(손)에 더 가까운 점 = 창을 따라 움직인 손
    is_hand = np.linalg.norm(flow, axis=1) < np.linalg.norm(flow - bg_flow, axis=1)
    hand, src = flow[is_hand], pts[is_hand]
    if len(hand) < MIN_HAND_POINTS:
        return None
    spread = float(np.median(np.linalg.norm(hand - np.median(hand, axis=0), axis=1)))
    return spread, affine_residual(src, src + hand)


def deformation(frames, centers: np.ndarray, moving: np.ndarray) -> tuple[float, int, float]:
    """이동 구간 쌍들의 (손 변형량 중앙값, 쓴 쌍 수, 변환 맞춤 오차 중앙값)."""
    vals, res = [], []
    for k in range(len(frames) - 1):
        if not (moving[k] and moving[k + 1]):
            continue
        d = centers[k + 1] - centers[k]
        d = np.array([-d[0], d[1]])     # crop은 좌우 반전되어 있으므로 x 방향을 뒤집는다
        if not MIN_MOVE <= np.abs(d).max() <= MAX_MOVE:
            continue
        v = pair_deform(frames[k], frames[k + 1], d)
        if v is not None:
            vals.append(v[0])
            res.append(v[1])
    if not vals:
        return float("nan"), 0, float("nan")
    return float(np.median(vals)), len(vals), float(np.nanmedian(res))


# ---------------------------------------------------------------- 특징

def extract(frames, raw_trajectory: np.ndarray, timestamps: np.ndarray, frame_size) -> FrameFeatures:
    raw = np.asarray(raw_trajectory, dtype=np.float64)
    centers = crop_centers(raw, frame_size)
    dwell = split_dwell(raw, np.asarray(timestamps, dtype=np.float64))
    zr, zn = temporal_noise(frames, centers, dwell)
    df, dn, ar = deformation(frames, centers, ~dwell)
    return FrameFeatures(zr, zn, df, dn, ar)


if __name__ == "__main__":
    # 사람 녹화 vs 합성 봇(S1~S4) 분포 비교
    import glob
    from pathlib import Path

    from cv.bot_detection.record import load_recording

    data_dir = Path(__file__).resolve().parents[1] / "test" / "data"
    files = sorted(glob.glob(str(data_dir / "human" / "*.npz"))) + \
        sorted(glob.glob(str(data_dir / "synth" / "S*" / "*.npz")))
    groups: dict[str, list[FrameFeatures]] = {}
    for f in files:
        d = load_recording(f)
        feat = extract(d["frames"], d["raw_trajectory"], d["timestamps"], tuple(d["frame_size"]))
        groups.setdefault(Path(f).parent.name, []).append(feat)

    for name in ("zero_ratio", "deform", "affine_res"):
        print(name)
        for g, feats in groups.items():
            v = np.array([getattr(x, name) for x in feats])
            ok = v[~np.isnan(v)]
            rng = f"{ok.min():.3f} ~ {ok.max():.3f} (median {np.median(ok):.3f})" if len(ok) else "-"
            print(f"  {g:6s} n={len(v):2d} 측정={len(ok):2d}  {rng}")
