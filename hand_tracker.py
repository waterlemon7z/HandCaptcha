"""손인식 모듈.

카메라 프레임에서 손을 찾아 문제 이미지(1280×720) 좌표로 바꾸고,
주먹을 쥐면 궤적 기록을 시작해 손이 1초 멈추면 정답 영역(box)을 확정한다.

구조
- 랜드마크 → (위치, 손 상태): palm_center(), classify_hand()   ← 순수 함수
- 상태 머신: HandTracker.update(point, hand_status, t)          ← 좌표만 넣어 테스트 가능
- 손 검출: HandTracker.process(frame)  (MediaPipe → process_landmarks → update)
- 화면: HandTracker.render(result)   (궤적은 그리지 않음)

좌표는 모두 문제 이미지 기준 (0~W-1, 0~H-1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# MediaPipe 손 랜드마크 번호
WRIST = 0
PALM_POINTS = (0, 5, 9, 13, 17)                      # 손바닥 중심 계산용 (손목 + 네 손가락 뿌리)
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))     # (손가락 끝, PIP 관절): 검지, 중지, 약지, 소지

# 상태 값
HAND_NONE, HAND_OPEN, HAND_FIST = "none", "open", "fist"
STATE_IDLE, STATE_RECORDING, STATE_DONE = "idle", "recording", "done"

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
KOREAN_FONT = Path("C:/Windows/Fonts/malgun.ttf")    # 힌트 패널 한글 글자용 (없으면 영어로 표시)


@dataclass
class HandResult:
    """process()가 매 프레임 돌려주는 결과."""

    position: Optional[tuple[int, int]]     # 문제 이미지 기준 손 좌표. 손이 없거나 문제 이미지 밖이면 None
    hand_status: str                        # "none" | "open" | "fist"
    state: str                              # "idle" | "recording" | "done"
    trajectory: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.int32))  # (N, 2) 움직인 경로 전체
    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))                 # (N,) 기록 시작부터의 초
    box: Optional[tuple[int, int, int, int]] = None      # done일 때 경로 전체를 감싸는 (x1, y1, x2, y2)
    stop_position: Optional[tuple[int, int]] = None      # done일 때 마지막에 멈춘 위치 (멈춘 1초 동안의 평균)


# ---------------------------------------------------------------- 랜드마크 → (위치, 상태)

def palm_center(landmarks) -> tuple[float, float]:
    """손바닥 중심 = 랜드마크 0, 5, 9, 13, 17의 평균 (정규화 좌표)."""
    pts = np.asarray(landmarks, dtype=float)[list(PALM_POINTS), :2]
    x, y = pts.mean(axis=0)
    return float(x), float(y)


def count_folded_fingers(landmarks) -> int:
    """엄지를 뺀 네 손가락 중 접힌 개수. 손가락 끝이 PIP 관절보다 손목에 가까우면 접힘."""
    pts = np.asarray(landmarks, dtype=float)[:, :2]
    wrist = pts[WRIST]
    folded = 0
    for tip, pip in FINGERS:
        if np.linalg.norm(pts[tip] - wrist) < np.linalg.norm(pts[pip] - wrist):
            folded += 1
    return folded


def classify_hand(landmarks, min_folded: int = 3) -> str:
    """접힌 손가락이 min_folded개 이상이면 "fist", 아니면 "open"."""
    return HAND_FIST if count_folded_fingers(landmarks) >= min_folded else HAND_OPEN


# ---------------------------------------------------------------- 손인식 클래스

class HandTracker:
    def __init__(
        self,
        problem_img: np.ndarray,
        hint_img: np.ndarray,
        *,
        stop_seconds: float = 1.0,     # 이 시간 동안
        stop_pixels: float = 15.0,     # 이동 범위가 이 값 미만이면 멈춘 것으로 보고 done
        lost_seconds: float = 0.5,     # recording 중 손이 이만큼 안 보이면 기록 취소
        alpha: float = 0.5,            # EMA 계수 (1.0이면 보정 없음)
        min_folded: int = 3,           # 주먹 판정: 접힌 손가락 수
        mirror: bool = True,           # 좌우 반전 (셀카처럼)
        panel_width: int = 320,        # 힌트 패널 폭
        model_path: str | Path = DEFAULT_MODEL,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha는 0 초과 1 이하여야 합니다")
        self.stop_seconds = stop_seconds
        self.stop_pixels = stop_pixels
        self.lost_seconds = lost_seconds
        self.alpha = alpha
        self.min_folded = min_folded
        self.mirror = mirror
        self.panel_width = panel_width
        self.model_path = Path(model_path)
        self._landmarker = None         # 첫 process() 때 만든다 (테스트는 MediaPipe 없이 동작)
        self._last_ts_ms = -1
        self.set_images(problem_img, hint_img)

    # ------------------------------------------------------------ 공개 API

    def set_images(self, problem_img: np.ndarray, hint_img: np.ndarray) -> None:
        """문제를 바꾼다. 상태도 초기화된다. 원본 배열은 수정하지 않는다."""
        self.problem_img = problem_img
        self.hint_img = hint_img
        self.height, self.width = problem_img.shape[:2]
        self._panel = self._build_hint_panel()
        self.reset()

    def reset(self) -> None:
        """다음 문제를 위해 초기화."""
        self.state = STATE_IDLE
        self._points: list[tuple[float, float]] = []
        self._times: list[float] = []
        self._box: Optional[tuple[int, int, int, int]] = None
        self._stop: Optional[tuple[int, int]] = None
        self._smoothed: Optional[tuple[float, float]] = None
        self._last_seen = 0.0

    def process(self, frame: np.ndarray) -> HandResult:
        """카메라 프레임(BGR) 한 장을 처리한다. 매 프레임 호출."""
        t = time.monotonic()
        return self.process_landmarks(self._detect(frame, t), t)

    def process_landmarks(self, landmarks, t: Optional[float] = None) -> HandResult:
        """손 랜드마크 21개(정규화 x, y) 또는 None으로 한 프레임을 처리한다."""
        if landmarks is None:
            return self.update(None, HAND_NONE, t)
        nx, ny = palm_center(landmarks)
        if self.mirror:
            nx = 1.0 - nx
        # 카메라 프레임 전체를 문제 이미지 영역에만 대응시킨다 (힌트 패널로는 매핑하지 않음)
        point = (nx * (self.width - 1), ny * (self.height - 1))
        return self.update(point, classify_hand(landmarks, self.min_folded), t)

    def update(self, point: Optional[tuple[float, float]], hand_status: str,
               t: Optional[float] = None) -> HandResult:
        """상태 머신. point는 문제 이미지 기준 좌표(보정 전), 손이 없으면 None."""
        if t is None:
            t = time.monotonic()

        if point is None:
            self._smoothed = None
        else:
            point = self._smooth(point)

        # 문제 이미지 밖(힌트 패널 쪽 포함)의 손은 없는 것으로 친다
        if point is not None and not self._inside(point):
            point = None

        if self.state == STATE_IDLE:
            if point is not None and hand_status == HAND_FIST:
                self.state = STATE_RECORDING
                self._add(point, t)
        elif self.state == STATE_RECORDING:
            if point is not None:
                self._add(point, t)
                if self._stopped(t):
                    self.state = STATE_DONE
                    self._box = self._make_box()
                    cx, cy = self._recent_points(t).mean(axis=0)
                    self._stop = (int(round(cx)), int(round(cy)))
            elif t - self._last_seen >= self.lost_seconds:
                # 손을 놓치면 궤적이 끊겨 영역을 믿을 수 없으므로 취소하고 처음부터
                self.state = STATE_IDLE
                self._points, self._times = [], []

        position = None if point is None else (int(round(point[0])), int(round(point[1])))
        return self._result(position, hand_status)

    def render(self, result: HandResult) -> np.ndarray:
        """사용자 화면: 왼쪽 문제 이미지(오버레이 포함) + 오른쪽 힌트 패널(항상 원본 그대로).

        궤적(이동 경로)과 경로 전체 box는 기록만 하고 화면에는 그리지 않는다.
        완료되면 마지막에 멈춘 위치만 초록 표시로 보여 준다.
        """
        left = self.problem_img.copy()
        color = {STATE_IDLE: (200, 200, 200), STATE_RECORDING: (0, 140, 255),
                 STATE_DONE: (0, 200, 0)}[result.state]

        if result.stop_position is not None:
            sx, sy = result.stop_position
            cv2.circle(left, (sx, sy), 24, (0, 200, 0), 3, cv2.LINE_AA)
            cv2.drawMarker(left, (sx, sy), (0, 200, 0), cv2.MARKER_CROSS, 20, 2)
        if result.position is not None:
            radius = 14 if result.hand_status == HAND_FIST else 10
            cv2.circle(left, result.position, radius, color, -1, cv2.LINE_AA)
            cv2.circle(left, result.position, radius, (0, 0, 0), 2, cv2.LINE_AA)

        text = f"{result.state.upper()}  hand: {result.hand_status}"
        cv2.rectangle(left, (0, 0), (420, 40), (0, 0, 0), -1)
        cv2.putText(left, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        return np.hstack([left, self._panel])

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # ------------------------------------------------------------ 내부

    def _smooth(self, p: tuple[float, float]) -> tuple[float, float]:
        if self._smoothed is None:
            self._smoothed = p
        else:
            a = self.alpha
            self._smoothed = (a * p[0] + (1 - a) * self._smoothed[0],
                              a * p[1] + (1 - a) * self._smoothed[1])
        return self._smoothed

    def _inside(self, p: tuple[float, float]) -> bool:
        return 0 <= p[0] <= self.width - 1 and 0 <= p[1] <= self.height - 1

    def _add(self, p: tuple[float, float], t: float) -> None:
        self._points.append(p)
        self._times.append(t)
        self._last_seen = t

    def _stopped(self, t: float) -> bool:
        """최근 stop_seconds 동안의 이동 범위(가로·세로 중 큰 값)가 stop_pixels 미만인가."""
        if t - self._times[0] < self.stop_seconds - 1e-9:
            return False   # 아직 1초치 기록이 없음
        recent = self._recent_points(t)
        spread = recent.max(axis=0) - recent.min(axis=0)
        return float(spread.max()) < self.stop_pixels

    def _recent_points(self, t: float) -> np.ndarray:
        """최근 stop_seconds 동안 기록된 점들."""
        times = np.asarray(self._times)
        return np.asarray(self._points)[times >= t - self.stop_seconds - 1e-9]

    def _make_box(self) -> tuple[int, int, int, int]:
        """궤적 전체를 감싸는 사각형. 문제 이미지 안으로 자르고 x1 < x2, y1 < y2를 보장."""
        pts = np.asarray(self._points)
        x1, y1 = np.floor(pts.min(axis=0)).astype(int)
        x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
        x1, x2 = np.clip([x1, x2], 0, self.width - 1)
        y1, y2 = np.clip([y1, y2], 0, self.height - 1)
        if x1 == x2:   # 폭이 0이면 1픽셀로 (이미지 끝이면 왼쪽으로)
            x1, x2 = (x1, x1 + 1) if x1 < self.width - 1 else (x1 - 1, x1)
        if y1 == y2:
            y1, y2 = (y1, y1 + 1) if y1 < self.height - 1 else (y1 - 1, y1)
        return int(x1), int(y1), int(x2), int(y2)

    def _result(self, position, hand_status) -> HandResult:
        traj = np.rint(np.asarray(self._points, dtype=float)).astype(np.int32).reshape(-1, 2)
        times = np.asarray(self._times, dtype=float)
        if len(times):
            times = times - times[0]
        done = self.state == STATE_DONE
        return HandResult(position, hand_status, self.state, traj, times,
                          self._box if done else None, self._stop if done else None)

    def _build_hint_panel(self) -> np.ndarray:
        """힌트 패널: 어두운 배경 + "찾을 대상" 글자 + 비율 유지해 축소·가운데 배치한 힌트 이미지."""
        h, w = self.height, self.panel_width
        panel = np.full((h, w, 3), 40, np.uint8)
        label_h, margin = 60, 20

        ih, iw = self.hint_img.shape[:2]
        scale = min((w - 2 * margin) / iw, (h - label_h - 2 * margin) / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        hint = cv2.resize(self.hint_img, (nw, nh), interpolation=interp)
        x0 = (w - nw) // 2
        y0 = label_h + (h - label_h - nh) // 2
        panel[y0:y0 + nh, x0:x0 + nw] = hint
        self.hint_rect = (x0, y0, nw, nh)   # 패널 안에서 힌트 이미지 위치 (테스트·디버그용)

        _draw_label(panel, "찾을 대상", "TARGET", (w // 2, label_h // 2))
        return panel

    def _detect(self, frame: np.ndarray, t: float):
        """MediaPipe로 손 랜드마크 21개 [(x, y), ...]를 찾는다. 손이 없으면 None."""
        import mediapipe as mp

        if self._landmarker is None:
            self._landmarker = self._create_landmarker()
        # VIDEO 모드는 타임스탬프가 반드시 증가해야 함
        ts = max(int(t * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not result.hand_landmarks:
            return None
        return [(lm.x, lm.y) for lm in result.hand_landmarks[0]]

    def _create_landmarker(self):
        # mediapipe 1.x에는 mp.solutions가 없어 Tasks API(HandLandmarker)를 쓴다
        from mediapipe.tasks.python import BaseOptions, vision

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"손 인식 모델 파일이 없습니다: {self.model_path}\n"
                "python download_model.py 로 내려받으세요.")
        options = vision.HandLandmarkerOptions(
            # Windows에서 MediaPipe(C++)가 한글 경로를 못 열어서 바이트로 넘긴다
            base_options=BaseOptions(model_asset_buffer=self.model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        return vision.HandLandmarker.create_from_options(options)


def _draw_label(img: np.ndarray, text: str, fallback: str, pos: tuple[int, int],
                align_left: bool = False) -> None:
    """한글 글자를 그린다 (pos는 글자 가운데, align_left면 글자 왼쪽 가운데).
    한글 폰트를 못 쓰면 영어(fallback)로."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(str(KOREAN_FONT), 28)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(pos, text, font=font, fill=(255, 255, 255),
                                 anchor="lm" if align_left else "mm")
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except (ImportError, OSError):
        (tw, th), _ = cv2.getTextSize(fallback, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x = pos[0] if align_left else pos[0] - tw // 2
        cv2.putText(img, fallback, (x, pos[1] + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
