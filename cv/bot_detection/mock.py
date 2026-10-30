"""진위검출용 봇 궤적 목데이터.

손인식(HandTracker)의 HandResult와 같은 형식으로 만든다.
- trajectory:     (N, 2) int32, 문제 이미지 픽셀 좌표 (0~W-1, 0~H-1), EMA 보정
- raw_trajectory: (N, 2) float64, trajectory와 같은 프레임의 EMA 전 원본 좌표 (반올림 안 함)
- timestamps:     (N,) float, 기록 시작부터의 초

봇이 좌표를 만들어도 손인식을 거쳐서 들어온다고 보고 똑같이 처리한다
(trajectory는 EMA + 반올림, raw_trajectory는 이미지 안으로 자르기만).
모든 궤적은 "이동 → 목표에서 멈춤"이고, 손인식이 done을 내는 프레임에서 잘린다
(최근 1초 동안 이동 범위 15px 미만이 되는 첫 프레임).

봇 종류
- linear:       등속 직선, 완전히 멈춤
- bezier:       베지어 곡선 + 최소 저크(부드러운 가감속), 완전히 멈춤
- noisy_linear: 등속 직선 + 매 프레임 가우시안 노이즈 (흔한 "사람 흉내")
- teleport:     시작점에서 목표로 한 번에 점프
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 1280, 720
FPS = 30.0
ALPHA = 0.5                     # 손인식 EMA 계수와 같게
DWELL_SECONDS = 1.3             # 손인식 stop_seconds(1.0)보다 길게 멈춤
KINDS = ("linear", "bezier", "noisy_linear", "teleport")


@dataclass
class MockTrajectory:
    kind: str
    trajectory: np.ndarray      # (N, 2) int32, EMA 보정
    timestamps: np.ndarray      # (N,) 초
    raw_trajectory: np.ndarray  # (N, 2) float64, EMA 전


# ---------------------------------------------------------------- 시간·점 생성

def make_timestamps(n: int, rng: np.random.Generator, fps: float = FPS,
                    jitter: float = 0.003) -> np.ndarray:
    """프레임 간격 1/fps에 웹캠 지터(초 단위 표준편차)를 더한 타임스탬프. 첫 값은 0."""
    dt = np.full(n - 1, 1.0 / fps) + rng.normal(0, jitter, n - 1)
    dt = np.clip(dt, 0.2 / fps, None)
    return np.concatenate([[0.0], np.cumsum(dt)])


def random_endpoints(rng: np.random.Generator, min_dist: float = 200.0,
                     margin: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """이미지 안에서 서로 min_dist 이상 떨어진 시작점·목표점."""
    lo, hi = np.array([margin, margin]), np.array([WIDTH - margin, HEIGHT - margin])
    while True:
        start, goal = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(goal - start) >= min_dist:
            return start, goal


def min_jerk(s: np.ndarray) -> np.ndarray:
    """0~1 진행률을 최소 저크 곡선으로 (양 끝에서 속도·가속도 0)."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def clip_to_image(points: np.ndarray) -> np.ndarray:
    return np.clip(points, [0, 0], [WIDTH - 1, HEIGHT - 1]).astype(np.float64)


def to_pixels(points: np.ndarray) -> np.ndarray:
    """이미지 안으로 자르고 int32로 반올림 (손인식 HandResult와 같게)."""
    return np.rint(clip_to_image(points)).astype(np.int32)


def apply_tracker_pipeline(points: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """손인식과 같은 처리. (EMA 보정한 trajectory, EMA 전 raw_trajectory)를 돌려준다."""
    smoothed = np.empty_like(points, dtype=float)
    smoothed[0] = points[0]
    for i in range(1, len(points)):
        smoothed[i] = alpha * points[i] + (1 - alpha) * smoothed[i - 1]
    return to_pixels(smoothed), clip_to_image(points)


# ---------------------------------------------------------------- 봇 종류별 경로 (보정 전 float)

def _n_frames(seconds: float, fps: float = FPS) -> int:
    return max(2, int(round(seconds * fps)))


def linear_path(rng: np.random.Generator) -> np.ndarray:
    start, goal = random_endpoints(rng)
    s = np.linspace(0, 1, _n_frames(rng.uniform(0.6, 1.5)))[:, None]
    move = start + s * (goal - start)
    dwell = np.repeat(goal[None], _n_frames(DWELL_SECONDS), axis=0)
    return np.vstack([move, dwell])


def bezier_path(rng: np.random.Generator) -> np.ndarray:
    start, goal = random_endpoints(rng)
    # 제어점: 직선에서 수직 방향으로 랜덤하게 휜다
    d = goal - start
    normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    c1 = start + d / 3 + normal * rng.uniform(-0.4, 0.4) * np.linalg.norm(d)
    c2 = start + 2 * d / 3 + normal * rng.uniform(-0.4, 0.4) * np.linalg.norm(d)
    s = min_jerk(np.linspace(0, 1, _n_frames(rng.uniform(0.6, 1.5))))[:, None]
    move = ((1 - s)**3 * start + 3 * (1 - s)**2 * s * c1
            + 3 * (1 - s) * s**2 * c2 + s**3 * goal)
    dwell = np.repeat(goal[None], _n_frames(DWELL_SECONDS), axis=0)
    return np.vstack([move, dwell])


def noisy_linear_path(rng: np.random.Generator, sigma: float = 2.0) -> np.ndarray:
    """백색 가우시안 노이즈. 사람 떨림(특정 주파수대)과 달리 전 주파수에 고르게 퍼진다."""
    path = linear_path(rng)
    return path + rng.normal(0, sigma, path.shape)


def teleport_path(rng: np.random.Generator) -> np.ndarray:
    start, goal = random_endpoints(rng)
    before = np.repeat(start[None], int(rng.integers(1, 4)), axis=0)
    after = np.repeat(goal[None], _n_frames(DWELL_SECONDS), axis=0)
    return np.vstack([before, after])


_PATHS = {
    "linear": linear_path,
    "bezier": bezier_path,
    "noisy_linear": noisy_linear_path,
    "teleport": teleport_path,
}


# ---------------------------------------------------------------- 공개 API

def make_bot(kind: str, rng: np.random.Generator | int | None = None) -> MockTrajectory:
    """봇 궤적 하나를 HandResult 형식으로 만든다."""
    if kind not in _PATHS:
        raise ValueError(f"알 수 없는 봇 종류: {kind} (가능: {KINDS})")
    rng = np.random.default_rng(rng)
    path = _PATHS[kind](rng)
    trajectory, raw = apply_tracker_pipeline(path)
    times = make_timestamps(len(path), rng)
    n = done_index(trajectory, times) + 1
    return MockTrajectory(kind, trajectory[:n], times[:n], raw[:n])


def done_index(trajectory: np.ndarray, times: np.ndarray, stop_seconds: float = 1.0,
               stop_pixels: float = 15.0) -> int:
    """손인식이 done을 내는 첫 프레임 (최근 stop_seconds 동안 이동 범위 < stop_pixels).
    손인식은 이 프레임에서 기록을 멈추므로 궤적도 여기서 끝난다."""
    for i in range(len(times)):
        if times[i] < stop_seconds - 1e-9:
            continue
        recent = trajectory[(times >= times[i] - stop_seconds - 1e-9) & (np.arange(len(times)) <= i)]
        if np.ptp(recent, axis=0).max() < stop_pixels:
            return i
    return len(times) - 1


def make_bots(n_per_kind: int = 10, seed: int = 0) -> list[MockTrajectory]:
    rng = np.random.default_rng(seed)
    return [make_bot(kind, rng) for kind in KINDS for _ in range(n_per_kind)]


def save_csv(mock: MockTrajectory, path: str | Path) -> None:
    """t,x,y,raw_x,raw_y CSV로 저장 (x,y = EMA 보정, raw_x,raw_y = EMA 전)."""
    data = np.column_stack([mock.timestamps, mock.trajectory, mock.raw_trajectory])
    np.savetxt(path, data, delimiter=",", header="t,x,y,raw_x,raw_y", comments="",
               fmt=["%.6f", "%d", "%d", "%.4f", "%.4f"])


def load_csv(path: str | Path) -> MockTrajectory:
    """save_csv로 저장한 CSV를 읽는다. kind는 파일 이름에서 번호를 뗀 것."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    kind = Path(path).stem.rsplit("_", 1)[0]
    return MockTrajectory(kind, data[:, 1:3].astype(np.int32), data[:, 0],
                          data[:, 3:5])


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "test" / "data" / "bot"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for m in make_bots():
        counts[m.kind] = counts.get(m.kind, 0) + 1
        save_csv(m, out_dir / f"{m.kind}_{counts[m.kind]:02d}.csv")
    print(f"{sum(counts.values())}개 저장: {out_dir}")
