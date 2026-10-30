"""합성 손 영상 봇 목데이터.

가상 카메라로 합성 영상을 넣는 봇을 흉내 낸다.
재료 사진(배경, 주먹, 검지)에서 손을 오려 배경 위로 경로대로 움직인 영상을 만들고,
그 영상을 손인식(HandTracker)에 실제로 통과시켜 사람 녹화와 같은 형식(.npz)으로 저장한다.

봇 수준
- S1: 손 사진을 그대로 붙여 이동, 배경 고정, 매끈한 경로
- S2: S1 + 경로 좌표에 부드러운 노이즈 (궤적 떨림 흉내)
- S3: S2 + 매 프레임 화면 전체에 가우시안 노이즈 (카메라 센서 노이즈 흉내)
- S4: S3 + 손 사진을 매 프레임 조금씩 회전·확대 (warpAffine, 손 변형 흉내)

재료: cv/test/data/synth/source/배경.jpg, 주먹.jpg, 검지1.jpg
실행: python -m cv.bot_detection.synth
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv.bot_detection.mock import min_jerk
from cv.bot_detection.record import dummy_images, save_result
from hand_tracker import DEFAULT_MODEL, INDEX_TIP, STATE_DONE, HandTracker

DATA_DIR = Path(__file__).resolve().parents[1] / "test" / "data" / "synth"
SOURCE_DIR = DATA_DIR / "source"
CAM_W, CAM_H = 640, 480          # 사람 녹화와 같은 카메라 크기
FPS = 30.0
FIST_FRAMES = 8                  # 처음 주먹 프레임 수 (기록 시작용)
MAX_SECONDS = 8.0                # 이 안에 done이 안 나면 실패로 본다
LEVELS = ("S1", "S2", "S3", "S4")
PATHS = ("linear", "bezier")


@dataclass
class Sprite:
    """오려낸 손. image/mask는 원본 크기, anchor는 검지 끝(사진 좌표)."""
    image: np.ndarray
    mask: np.ndarray            # uint8 0/255
    anchor: np.ndarray          # (x, y)


# ---------------------------------------------------------------- 재료 준비

def read_image(path: Path) -> np.ndarray:
    """한글 경로도 읽히게 imdecode로 읽고, 4:3 가운데를 잘라 CAM_W×CAM_H로 맞춘다."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    cw = min(w, h * 4 // 3)
    x0 = (w - cw) // 2
    return cv2.resize(img[:, x0:x0 + cw], (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)


def detect_landmarks(img: np.ndarray) -> np.ndarray:
    """MediaPipe(IMAGE 모드)로 손 랜드마크 21개를 픽셀 좌표 (21, 2)로."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=DEFAULT_MODEL.read_bytes()),
        running_mode=vision.RunningMode.IMAGE, num_hands=1)
    with vision.HandLandmarker.create_from_options(options) as lm:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        raise RuntimeError("재료 사진에서 손을 찾지 못했습니다")
    h, w = img.shape[:2]
    return np.array([(p.x * (w - 1), p.y * (h - 1)) for p in result.hand_landmarks[0]])


def cut_sprite(img: np.ndarray, grow: int = 18) -> Sprite:
    """랜드마크를 감싸는 다각형(convexHull)을 grow 픽셀 팽창시킨 마스크로 손을 오린다."""
    pts = detect_landmarks(img)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 255)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    return Sprite(img, mask, pts[INDEX_TIP])


# ---------------------------------------------------------------- 영상 생성

def paste(bg: np.ndarray, sprite: Sprite, tip: np.ndarray,
          angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """검지 끝이 tip에 오도록 손을 붙인다. angle(도)·scale은 검지 끝 기준 회전·확대."""
    m = cv2.getRotationMatrix2D(tuple(map(float, sprite.anchor)), angle, scale)
    m[:, 2] += tip - sprite.anchor
    size = (bg.shape[1], bg.shape[0])
    img = cv2.warpAffine(sprite.image, m, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(sprite.mask, m, size, flags=cv2.INTER_NEAREST)
    out = bg.copy()
    out[mask > 0] = img[mask > 0]
    return out


def make_path(kind: str, rng: np.random.Generator) -> np.ndarray:
    """카메라 좌표의 검지 끝 경로 (이동 + 멈춤). 멈춤은 done이 날 때까지 충분히 길게."""
    margin = np.array([120, 90])
    lo, hi = margin, np.array([CAM_W, CAM_H]) - margin
    while True:
        start, goal = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(goal - start) >= 150:
            break
    n = max(2, int(round(rng.uniform(0.6, 1.5) * FPS)))
    s = np.linspace(0, 1, n)
    if kind == "linear":
        move = start + s[:, None] * (goal - start)
    else:
        d = goal - start
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        c1 = start + d / 3 + normal * rng.uniform(-0.4, 0.4) * np.linalg.norm(d)
        c2 = start + 2 * d / 3 + normal * rng.uniform(-0.4, 0.4) * np.linalg.norm(d)
        u = min_jerk(s)[:, None]
        move = (1 - u)**3 * start + 3 * (1 - u)**2 * u * c1 + 3 * (1 - u) * u**2 * c2 + u**3 * goal
    dwell = np.repeat(goal[None], int(MAX_SECONDS * FPS), axis=0)
    return np.vstack([np.repeat(start[None], FIST_FRAMES, axis=0), move, dwell])


def smooth_noise(n: int, sigma: float, rng: np.random.Generator, corr: float = 2.0) -> np.ndarray:
    """이웃 프레임끼리 이어진 (N, 2) 노이즈. 백색 노이즈를 가우시안으로 흐린 뒤 표준편차를 sigma로."""
    white = rng.normal(0, 1, (n, 1, 2)).astype(np.float32)
    k = cv2.getGaussianKernel(int(6 * corr) | 1, corr, cv2.CV_32F)
    out = cv2.sepFilter2D(white, -1, np.ones((1, 1), np.float32), k,
                          borderType=cv2.BORDER_REFLECT).reshape(n, 2)
    return out / (out.std(axis=0) + 1e-9) * sigma


def make_frames(level: str, kind: str, bg: np.ndarray, fist: Sprite, point: Sprite,
                rng: np.random.Generator):
    """봇 영상 프레임을 하나씩 만든다 (제너레이터)."""
    path = make_path(kind, rng)
    n = len(path)
    if level in ("S2", "S3", "S4"):
        path = path + smooth_noise(n, 1.0, rng)
    angles = smooth_noise(n, 2.0, rng, corr=4.0)[:, 0] if level == "S4" else np.zeros(n)
    scales = 1 + smooth_noise(n, 0.02, rng, corr=4.0)[:, 1] if level == "S4" else np.ones(n)
    for i in range(n):
        sprite = fist if i < FIST_FRAMES else point
        frame = paste(bg, sprite, path[i], angles[i], scales[i])
        if level in ("S3", "S4"):
            noise = rng.normal(0, 2.0, frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        yield frame


def run_tracker(frames, rng: np.random.Generator):
    """프레임을 손인식에 넣는다. 시각은 1/FPS 간격 + 웹캠 지터. done 결과 또는 None."""
    tracker = HandTracker(*dummy_images())
    t = 0.0
    try:
        for frame in frames:
            result = tracker.process_landmarks(tracker._detect(frame, t), t, frame)
            if result.state == STATE_DONE:
                return result
            t += 1 / FPS + rng.normal(0, 0.003)
    finally:
        tracker.close()
    return None


def main(n_per_kind: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    bg = read_image(SOURCE_DIR / "배경.jpg")
    fist = cut_sprite(read_image(SOURCE_DIR / "주먹.jpg"))
    point = cut_sprite(read_image(SOURCE_DIR / "검지1.jpg"))
    # cv2.imwrite는 한글 경로에 조용히 실패해서 imencode + tofile로 저장
    preview = paste(bg, point, np.array([CAM_W / 2, CAM_H / 2]))
    cv2.imencode(".jpg", preview)[1].tofile(str(DATA_DIR / "preview.jpg"))

    for level in LEVELS:
        out_dir = DATA_DIR / level
        for kind in PATHS:
            for i in range(n_per_kind):
                result = run_tracker(make_frames(level, kind, bg, fist, point, rng), rng)
                name = f"{level}_{kind}_{i + 1:02d}"
                if result is None:
                    print(f"{name}: done 안 됨 (손 인식 실패 또는 멈춤 판정 안 남)")
                    continue
                save_result(result, out_dir, name)
                print(f"{name}: 점 {len(result.timestamps)}개, {result.timestamps[-1]:.2f}초")


if __name__ == "__main__":
    main()
