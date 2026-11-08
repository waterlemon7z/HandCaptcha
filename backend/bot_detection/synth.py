"""합성 손 영상 봇 목데이터 (2026-11-07 개정: 1280×720, 뼈대 마스크, 손 크기 맞춤, 긴 이동).

가상 카메라로 합성 영상을 넣는 봇을 흉내 낸다.
재료 사진에서 손을 오려 배경 위로 경로대로 움직인 영상을 만들고,
그 영상을 손인식(HandTracker)에 실제로 통과시켜 사람 녹화와 같은 형식(.npz)으로 저장한다.

봇 수준
- S1: 손 사진을 그대로 붙여 이동, 배경 고정, 매끈한 경로
- S2: S1 + 경로 좌표에 부드러운 노이즈 (궤적 떨림 흉내)
- S3: S2 + 매 프레임 화면 전체에 가우시안 노이즈 (카메라 센서 노이즈 흉내)
- S4: S3 + 손 사진을 매 프레임 조금씩 회전·확대 (warpAffine, 손 변형 흉내)

개정 내용 (예전 S1~S4는 synth/old_640x480/에 보관)
- 카메라 1280×720 (앱 Camera와 같음)
- 마스크: 랜드마크 convexHull(검지와 주먹 사이 배경까지 잘림 → 손 둘레에 배경 테두리가 티 남) 대신
  손 뼈대(손바닥 다각형 + 손가락 마디마다 굵은 선) + 팔뚝 띠로 오리고, 경계는 가우시안으로 부드럽게 섞음 (03강 블렌딩)
  (팔 없이 손목에서 자르면 1280×720에서 MediaPipe가 손을 거의 못 찾음)
- 영상마다 배경·손 사진·크기 조합을 한 장으로 미리 확인해서, 손 인식이 되는 조합만 씀
- 손 크기: 사람 녹화처럼 손끝이 crop을 채우도록 손 크기(손목~중지 뿌리)를 HAND_PX 범위로 맞춤
- 이동: 1.5~3초, 300px 이상 (CNN 10장 묶음이 나오게)
- 재료: 검지 사진 여러 장(source/검지*.jpg) × 배경 여러 장(source/배경*.jpg + assets/backgrounds/*.jpg)

재료: backend/bot_detection/test/data/synth/source/ (배경*.jpg, 주먹.jpg, 검지*.jpg)
실행: python -m backend.bot_detection.synth            (종류별 20개)
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.bot_detection.mock import min_jerk
from backend.bot_detection.record import dummy_images, save_result
from cv.hand_tracking.hand_tracker import DEFAULT_MODEL, INDEX_TIP, STATE_DONE, HandTracker

DATA_DIR = Path(__file__).resolve().parent / "test" / "data" / "synth"
SOURCE_DIR = DATA_DIR / "source"
ASSET_BG_DIR = Path(__file__).resolve().parents[2] / "assets" / "backgrounds"
CAM_W, CAM_H = 1280, 720         # 앱 Camera와 같은 카메라 크기
FPS = 30.0
FIST_FRAMES = 8                  # 처음 주먹 프레임 수 (기록 시작용: 주먹 → 검지 펴기)
MAX_SECONDS = 8.0                # 이 안에 done이 안 나면 실패로 본다
LEVELS = ("S1", "S2", "S3", "S4")
PATHS = ("linear", "bezier")
HAND_PX = (170.0, 260.0)         # 손 크기(손목~중지 뿌리, 카메라 px) 범위. 사람 녹화에서 손끝이 128 crop을 채우는 정도
MOVE_SECONDS = (1.5, 3.0)        # 이동 시간
MIN_DIST = 300.0                 # 시작~끝 최소 거리 (카메라 px)
GROW = 0.06                      # 뼈대 마스크를 손 크기 × 이만큼 넓힘
ARM_WIDTH, ARM_LEN = 0.9, 4.0    # 팔뚝 띠: 폭 = 손 크기 × 0.9, 길이 = 손 크기 × 4
START_TIMEOUT = 40               # 이 프레임까지 기록이 시작 안 되면 포기 (손 인식 실패)
MAX_TRIES = 30                   # 영상 하나당 배경·손 사진·크기 조합을 바꿔 다시 시도하는 횟수 (미리 확인은 한 장이라 빠름)

WRIST, MIDDLE_MCP = 0, 9
PALM = (0, 1, 5, 9, 13, 17)                                  # 손바닥 다각형
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


@dataclass
class Sprite:
    """오려낸 손. image/alpha는 원본 크기, anchor는 검지 끝(사진 좌표), hand는 손 크기(px)."""
    image: np.ndarray
    alpha: np.ndarray           # float32 0~1 (경계 부드럽게)
    anchor: np.ndarray          # (x, y)
    hand: float


# ---------------------------------------------------------------- 재료 준비

def read_image(path: Path) -> np.ndarray:
    """한글 경로도 읽히게 imdecode로 읽고, 16:9 가운데를 잘라 CAM_W×CAM_H로 맞춘다."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if w * CAM_H > h * CAM_W:
        cw = h * CAM_W // CAM_H
        img = img[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = w * CAM_H // CAM_W
        img = img[(h - ch) // 2:(h - ch) // 2 + ch]
    return cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)


_IMAGE_LM = None


def _image_landmarker():
    """MediaPipe IMAGE 모드 손 검출기 (한 번 만들어 재사용)."""
    global _IMAGE_LM
    if _IMAGE_LM is None:
        from mediapipe.tasks.python import BaseOptions, vision
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=DEFAULT_MODEL.read_bytes()),
            running_mode=vision.RunningMode.IMAGE, num_hands=1)
        _IMAGE_LM = vision.HandLandmarker.create_from_options(options)
    return _IMAGE_LM


def detect_landmarks(img: np.ndarray) -> np.ndarray:
    """MediaPipe(IMAGE 모드)로 손 랜드마크 21개를 픽셀 좌표 (21, 2)로."""
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = _image_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        raise RuntimeError("재료 사진에서 손을 찾지 못했습니다")
    h, w = img.shape[:2]
    return np.array([(p.x * (w - 1), p.y * (h - 1)) for p in result.hand_landmarks[0]])


def skeleton_mask(shape, pts: np.ndarray, hand: float) -> np.ndarray:
    """손 뼈대 마스크 (uint8 0/255): 손바닥 다각형 + 손가락 마디마다 손가락 굵기의 선.
    convexHull과 달리 펴진 검지와 접힌 손가락 사이의 배경은 안 들어간다."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(pts[list(PALM)]).astype(np.int32)], 255)
    width = max(3, int(round(hand * 0.22)))                       # 손가락 굵기 ≈ 손 크기 × 0.22
    for chain in FINGERS:
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(mask, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), 255, width)
        tip = tuple(np.round(pts[chain[-1]]).astype(int))
        cv2.circle(mask, tip, width // 2, 255, -1)               # 손끝 둥글게
    cv2.line(mask, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)), 255, width)
    # 팔뚝: 손목에서 (손목 − 중지 뿌리) 방향으로 띠 모양. 재료 사진에 팔이 이어져 있고,
    # 손목에서 잘라 손만 붙이면 MediaPipe가 손으로 인식을 못 함 (실제 공격자도 팔까지 오림)
    wrist = pts[WRIST]
    d = wrist - pts[MIDDLE_MCP]
    d = d / (np.linalg.norm(d) + 1e-9)
    nrm = np.array([-d[1], d[0]])
    w0, w1 = hand * ARM_WIDTH / 2, hand * ARM_WIDTH * 0.85 / 2
    end = wrist + d * hand * ARM_LEN
    arm = np.array([wrist + nrm * w0, wrist - nrm * w0, end - nrm * w1, end + nrm * w1])
    cv2.fillPoly(mask, [np.round(arm).astype(np.int32)], 255)
    # 아주 살짝 넓힘: 너무 딱 맞게 오리면 복잡한 배경 위에서 손 윤곽이 묻혀 MediaPipe가 손을 못 찾음
    r = max(1, int(round(hand * GROW)))
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def cut_sprite(img: np.ndarray) -> Sprite:
    pts = detect_landmarks(img)
    hand = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))
    mask = skeleton_mask(img.shape, pts, hand)
    alpha = cv2.GaussianBlur(mask, (0, 0), max(1.0, hand * 0.02)).astype(np.float32) / 255.0
    return Sprite(img, alpha, pts[INDEX_TIP], hand)


def load_backgrounds() -> list[np.ndarray]:
    files = sorted(glob.glob(str(SOURCE_DIR / "배경*.jpg"))) + sorted(glob.glob(str(ASSET_BG_DIR / "*.jpg")))
    return [read_image(Path(f)) for f in files]


# ---------------------------------------------------------------- 영상 생성

def paste(bg: np.ndarray, sprite: Sprite, tip: np.ndarray, angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """검지 끝이 tip에 오도록 손을 붙인다 (알파 블렌딩). angle(도)·scale은 검지 끝 기준 회전·확대."""
    m = cv2.getRotationMatrix2D(tuple(map(float, sprite.anchor)), angle, scale)
    m[:, 2] += tip - sprite.anchor
    size = (bg.shape[1], bg.shape[0])
    img = cv2.warpAffine(sprite.image, m, size, flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(sprite.alpha, m, size, flags=cv2.INTER_LINEAR)[..., None]
    return (a * img + (1 - a) * bg).astype(np.uint8)


def make_path(kind: str, rng: np.random.Generator) -> np.ndarray:
    """카메라 좌표의 검지 끝 경로 (주먹 + 이동 + 멈춤). 멈춤은 done이 날 때까지 충분히 길게."""
    margin = np.array([220, 160])
    lo, hi = margin, np.array([CAM_W, CAM_H]) - margin
    while True:
        start, goal = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(goal - start) >= MIN_DIST:
            break
    n = max(2, int(round(rng.uniform(*MOVE_SECONDS) * FPS)))
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


def detectable(bg: np.ndarray, fist: Sprite, point: Sprite, target: float) -> bool:
    """이 배경·손 사진·크기 조합에서 손 인식이 주먹과 검지를 둘 다 찾는가 (한 장씩 미리 확인)."""
    tip = np.array([CAM_W / 2, CAM_H / 2])
    for sp in (fist, point):
        try:
            detect_landmarks(paste(bg, sp, tip, 0.0, target / sp.hand))
        except RuntimeError:
            return False
    return True


def make_frames(level: str, kind: str, bg: np.ndarray, fist: Sprite, point: Sprite,
                rng: np.random.Generator, target: float):
    """봇 영상 프레임을 하나씩 만든다 (제너레이터). target = 손 크기(px)."""
    path = make_path(kind, rng)
    n = len(path)
    base_point, base_fist = target / point.hand, target / fist.hand
    if level in ("S2", "S3", "S4"):
        path = path + smooth_noise(n, 2.0, rng)                     # 640 기준 1px → 1280 기준 2px
    angles = smooth_noise(n, 2.0, rng, corr=4.0)[:, 0] if level == "S4" else np.zeros(n)
    scales = 1 + smooth_noise(n, 0.02, rng, corr=4.0)[:, 1] if level == "S4" else np.ones(n)
    for i in range(n):
        if i < FIST_FRAMES:
            frame = paste(bg, fist, path[i], 0.0, base_fist)
        else:
            frame = paste(bg, point, path[i], angles[i], base_point * scales[i])
        if level in ("S3", "S4"):
            noise = rng.normal(0, 2.0, frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        yield frame


def run_tracker(frames, rng: np.random.Generator):
    """프레임을 손인식에 넣는다. 시각은 1/FPS 간격 + 웹캠 지터. done 결과 또는 None."""
    tracker = HandTracker(*dummy_images())
    t = 0.0
    try:
        for i, frame in enumerate(frames):
            result = tracker.process_landmarks(tracker._detect(frame, t), t, frame)
            if result is not None and result.state == STATE_DONE:
                return result
            if i >= START_TIMEOUT and (result is None or result.state == "idle"):
                return None                                         # 기록이 시작도 안 됨 → 일찍 포기
            t += 1 / FPS + rng.normal(0, 0.003)
    finally:
        tracker.close()
    return None


def main(n_per_kind: int = 10, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    bgs = load_backgrounds()
    fist = cut_sprite(read_image(SOURCE_DIR / "주먹.jpg"))
    points = [cut_sprite(read_image(Path(f))) for f in sorted(glob.glob(str(SOURCE_DIR / "검지*.jpg")))]
    print(f"재료: 배경 {len(bgs)}장, 검지 사진 {len(points)}장 (손 크기 {[round(p.hand) for p in points]}px), "
          f"주먹 손 크기 {fist.hand:.0f}px")
    # 미리보기: 배경 위에 손 하나 (cv2.imwrite는 한글 경로에 조용히 실패해서 imencode + tofile)
    preview = paste(bgs[-1], points[0], np.array([CAM_W / 2, CAM_H / 2]), 0.0, np.mean(HAND_PX) / points[0].hand)
    cv2.imencode(".jpg", preview)[1].tofile(str(DATA_DIR / "preview.jpg"))

    for level in LEVELS:
        out_dir = DATA_DIR / level
        for kind in PATHS:
            for i in range(n_per_kind):
                name = f"{level}_{kind}_{i + 1:02d}"
                result = None
                for attempt in range(MAX_TRIES):
                    bg = bgs[int(rng.integers(len(bgs)))]
                    point = points[int(rng.integers(len(points)))]
                    target = rng.uniform(*HAND_PX)
                    if not detectable(bg, fist, point, target):
                        continue                                    # 이 조합은 손 인식이 못 찾음 → 다른 조합
                    result = run_tracker(make_frames(level, kind, bg, fist, point, rng, target), rng)
                    if result is not None:
                        break
                if result is None:
                    print(f"{name}: {MAX_TRIES}번 시도해도 done 안 됨 (손 인식 실패)", flush=True)
                    continue
                save_result(result, out_dir, name)
                print(f"{name}: 점 {len(result.timestamps)}개, {result.timestamps[-1]:.2f}초 (시도 {attempt + 1})", flush=True)


if __name__ == "__main__":
    main()
