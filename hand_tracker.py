"""HandCAPTCHA 손 인식 모듈.

웹캠 프레임에서 검지 끝 좌표를 뽑아 문제 캔버스 좌표로 바꾸고,
EMA 스무딩 → dwell 선택 → 궤적 기록까지 처리한다.

좌표 기준
- 캔버스: 왼쪽 이미지 | gap | 오른쪽 이미지 를 가로로 붙인 화면 (원본 픽셀 크기 그대로)
  크기 = (2 * W + gap, H)
- HandState.point, 궤적: 캔버스 좌표
- HandState.answer: 짚은 쪽 이미지의 원본 기준 (x, y)  → Captcha.solve(img_hash, answer)

웹캠 없이 테스트할 때는 update(frame) 대신 update_point(norm_xy, t)에
정규화 좌표(0~1, 이미 좌우 반전된 값)를 직접 넣으면 된다.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields, replace
from typing import Callable, Optional, Protocol

Point = tuple[float, float]
TrajectoryPoint = tuple[float, float, float]  # (x, y, t초)

# HandState.status 값 (GUI에서 원하는 문구로 바꿔 표시)
STATUS_NO_HAND = "NO_HAND"      # 손이 보이지 않음
STATUS_TRACKING = "TRACKING"    # 손끝 추적 중 (dwell 진행 없음)
STATUS_DWELLING = "DWELLING"    # 한 자리에 머무는 중 (게이지 차는 중)
STATUS_SELECTED = "SELECTED"    # 선택 확정 (reset() 전까지 잠김)


@dataclass(frozen=True)
class HandConfig:
    """손 인식 설정값."""

    smoothing: float = 0.4          # EMA 계수 α. 클수록 현재 값을 많이 반영 (1.0이면 스무딩 없음)
    dwell_radius: float = 30.0      # dwell 반경 (캔버스 픽셀)
    dwell_time: float = 1.0         # 반경 안에 머물러야 하는 시간 (초)
    active_margin: float = 0.075    # 카메라 각 변에서 버리는 비율. 0.075면 중앙 85%가 캔버스 전체에 대응
    lost_grace: float = 0.15        # 손을 놓쳐도 이 시간(초)까지는 상태 유지 (검출이 잠깐 튀는 것 대비)
    mirror: bool = True             # 웹캠 좌우 반전 (거울처럼 보이게)
    model_path: str = "models/hand_landmarker.task"  # 상대 경로면 이 파일 위치 기준
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class HandState:
    """매 프레임 update()가 돌려주는 상태."""

    point: Optional[Point]          # 스무딩된 손끝 캔버스 좌표. 손이 없으면 None
    progress: float                 # dwell 진행률 0~1
    selected: bool                  # 선택 확정 여부
    answer: Optional[tuple[int, int]]  # 확정 시 원본 이미지 기준 (x, y), 아니면 None
    side: Optional[str]             # 확정 시 "left" / "right"
    hand_visible: bool
    status: str


class FingertipDetector(Protocol):
    """손끝 검출기 인터페이스. 테스트에서는 가짜 검출기로 바꿔 끼운다."""

    def detect(self, frame, timestamp_ms: int) -> Optional[Point]:
        """정규화 좌표 (0~1, 좌우 반전 반영)를 돌려준다. 손이 없으면 None."""
        ...


class EMASmoother:
    """지수 이동 평균: s = α·p + (1-α)·s."""

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("smoothing(α)은 0 초과 1 이하여야 합니다")
        self.alpha = alpha
        self._value: Optional[Point] = None

    def update(self, p: Point) -> Point:
        if self._value is None:
            self._value = p
        else:
            a = self.alpha
            sx, sy = self._value
            self._value = (a * p[0] + (1 - a) * sx, a * p[1] + (1 - a) * sy)
        return self._value

    def reset(self) -> None:
        self._value = None


class DwellSelector:
    """반경 안에 일정 시간 머물면 선택을 확정한다.

    반경 안에 처음 들어온 점을 기준점으로 잡고, 기준점에서 반경 밖으로 나가면
    그 점을 새 기준점으로 다시 시작한다. 확정 좌표는 머무는 동안 점들의 평균.
    """

    def __init__(self, radius: float, dwell_time: float):
        if radius <= 0 or dwell_time <= 0:
            raise ValueError("dwell_radius와 dwell_time은 0보다 커야 합니다")
        self.radius = radius
        self.dwell_time = dwell_time
        self.reset()

    def reset(self) -> None:
        self._anchor: Optional[Point] = None
        self._start = 0.0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._count = 0

    def update(self, p: Point, t: float) -> float:
        """점을 추가하고 진행률(0~1)을 돌려준다."""
        if self._anchor is None or math.dist(p, self._anchor) > self.radius:
            self._anchor = p
            self._start = t
            self._sum_x = self._sum_y = 0.0
            self._count = 0
        self._sum_x += p[0]
        self._sum_y += p[1]
        self._count += 1
        elapsed = t - self._start
        # 부동소수점 오차로 정확히 dwell_time이 된 프레임을 놓치지 않도록 작은 여유를 둔다
        if elapsed >= self.dwell_time - 1e-9:
            return 1.0
        return elapsed / self.dwell_time

    def mean_point(self) -> Optional[Point]:
        if self._count == 0:
            return None
        return (self._sum_x / self._count, self._sum_y / self._count)


class HandTracker:
    """손끝 추적 + dwell 선택 + 궤적 기록.

    사용 예:
        tracker = HandTracker((W, H), gap=20, smoothing=0.4)
        state = tracker.update(frame)        # 매 프레임 (웹캠 원본 BGR 프레임)
        if state.selected:
            ok = captcha.solve(img_hash, state.answer)
            trajectory = tracker.get_trajectory()
        tracker.reset()                      # 새 문제 시작
    """

    def __init__(
        self,
        target_size: tuple[int, int],
        config: Optional[HandConfig] = None,
        *,
        gap: int = 0,
        detector: Optional[FingertipDetector] = None,
        clock: Callable[[], float] = time.monotonic,
        **overrides,
    ):
        """
        target_size: 문제 이미지 한 장의 원본 크기 (W, H)
        config: 설정값. overrides(smoothing=0.3 등)로 일부만 바꿀 수도 있음
        gap: GUI에서 좌우 이미지 사이에 둔 간격 (픽셀)
        detector: 손끝 검출기. None이면 첫 update(frame) 때 MediaPipe 검출기를 만든다
        clock: 현재 시각(초)을 돌려주는 함수
        """
        valid = {f.name for f in fields(HandConfig)}
        unknown = set(overrides) - valid
        if unknown:
            raise TypeError(f"알 수 없는 설정값: {sorted(unknown)}")
        self.config = replace(config or HandConfig(), **overrides)

        self.image_w, self.image_h = int(target_size[0]), int(target_size[1])
        self.gap = int(gap)
        self.canvas_size = (2 * self.image_w + self.gap, self.image_h)

        self._detector = detector
        self._clock = clock
        self._smoother = EMASmoother(self.config.smoothing)
        self._dwell = DwellSelector(self.config.dwell_radius, self.config.dwell_time)
        self.reset()

    # ------------------------------------------------------------------ 공개 API

    def update(self, frame) -> HandState:
        """웹캠 프레임 한 장을 처리한다 (BGR, 좌우 반전하지 않은 원본)."""
        raise NotImplementedError("MediaPipe 검출기 연결 전")

    def update_point(self, norm_xy: Optional[Point], t: Optional[float] = None) -> HandState:
        """정규화 손끝 좌표로 한 프레임을 처리한다.

        norm_xy: (0~1, 0~1) 좌우 반전이 반영된 좌표. 손이 없으면 None
        t: 시각(초). None이면 clock()
        """
        if t is None:
            t = self._clock()
        if self._t0 is None:
            self._t0 = t
        rel = t - self._t0

        # 선택이 확정되면 reset() 전까지 상태를 고정 (궤적도 더 쌓지 않음)
        if self._selected:
            return self._state

        if norm_xy is None:
            return self._on_hand_missing(rel)

        p = self._smoother.update(self.normalized_to_canvas(norm_xy))
        self._trajectory.append((p[0], p[1], rel))
        self._last_seen = rel

        progress = self._dwell.update(p, rel)
        if progress >= 1.0:
            answer_pt = self._dwell.mean_point() or p
            side, ix, iy = self.canvas_to_image(answer_pt)
            self._selected = True
            self._state = HandState(p, 1.0, True, (ix, iy), side, True, STATUS_SELECTED)
        else:
            status = STATUS_DWELLING if progress > 0 else STATUS_TRACKING
            self._state = HandState(p, progress, False, None, None, True, status)
        return self._state

    def get_trajectory(self) -> list[TrajectoryPoint]:
        """reset() 이후 기록된 궤적 [(x, y, t), ...]. t는 reset 후 첫 프레임 기준 초."""
        return list(self._trajectory)

    def reset(self) -> None:
        """새 문제를 시작할 때 호출. 스무딩, dwell, 궤적, 선택 상태를 모두 비운다."""
        self._smoother.reset()
        self._dwell.reset()
        self._trajectory: list[TrajectoryPoint] = []
        self._t0: Optional[float] = None
        self._last_seen: Optional[float] = None
        self._selected = False
        self._state = HandState(None, 0.0, False, None, None, False, STATUS_NO_HAND)

    @property
    def state(self) -> HandState:
        """가장 최근 상태."""
        return self._state

    # ------------------------------------------------------------------ 좌표 변환

    def normalized_to_canvas(self, norm_xy: Point) -> Point:
        """정규화 좌표 → 캔버스 픽셀 좌표. 카메라 중앙 활성 영역을 캔버스 전체에 늘린다."""
        m = self.config.active_margin
        span = 1.0 - 2.0 * m
        cw, ch = self.canvas_size
        u = min(1.0, max(0.0, (norm_xy[0] - m) / span))
        v = min(1.0, max(0.0, (norm_xy[1] - m) / span))
        return (u * (cw - 1), v * (ch - 1))

    def canvas_to_image(self, p: Point) -> tuple[str, int, int]:
        """캔버스 좌표 → (짚은 쪽, 원본 이미지 x, y). gap 위의 점은 가까운 쪽 이미지 가장자리로 붙인다."""
        x, y = p
        w = self.image_w
        if x < w + self.gap / 2:
            side, ix = "left", x
        else:
            side, ix = "right", x - (w + self.gap)
        ix = min(w - 1, max(0, round(ix)))
        iy = min(self.image_h - 1, max(0, round(y)))
        return side, ix, iy

    # ------------------------------------------------------------------ 내부

    def _on_hand_missing(self, rel: float) -> HandState:
        if self._last_seen is not None and rel - self._last_seen <= self.config.lost_grace:
            # 잠깐 놓친 것은 무시하고 직전 상태 유지 (궤적에는 기록하지 않음)
            return self._state
        self._smoother.reset()
        self._dwell.reset()
        self._state = HandState(None, 0.0, False, None, None, False, STATUS_NO_HAND)
        return self._state
