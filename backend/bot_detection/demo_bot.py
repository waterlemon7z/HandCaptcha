"""봇 시연용 영상 만들기 (발표 데모 전용).

시드를 고정하면 문제(정답 위치)가 항상 같다. 그 정답 위치로 손 사진을 움직이는
합성 손 영상(위협 모델의 "가상 카메라 봇")을 미리 만들어 두고, 앱이 웹캠 대신 이 영상을 읽게 한다.

순서
1. ProblemGenerator(validator=DifficultyCheck())로 seed 문제를 만들고 정답 박스 중심을 얻는다 (앱과 같은 ORB/SIFT 검수)
2. synth.py 재료(주먹·검지 사진, 배경)로 한 사이클 영상을 만든다:
   손 없음 → 주먹 → 검지 펴고 정답까지 이동 → 멈춤. 앱은 이 영상을 끝없이 반복 재생한다
   (문제가 준비되기 전에 지나간 사이클은 무시되고, 다음 사이클의 주먹부터 인식된다)
3. 저장한 영상을 다시 읽어 앱과 똑같이 손 인식 → 정답 판정 → 진위 검출까지 돌려 본다 (미리 검증)

앱의 Camera는 읽은 프레임을 좌우 반전하므로, 영상에는 반전한 프레임을 저장한다.
영상은 레포 바깥 폴더(main.py의 BOT_VIDEO_DIR)에 bot_<seed>.avi로 저장한다 (웹캠 영상이라 git에 안 넣음).

실행 (레포 루트, .venv):
    python -m backend.bot_detection.demo_bot --seed 1234
시연:
    python main.py --bot 1234
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from backend.bot_detection import detector
from backend.bot_detection.mock import min_jerk
from backend.bot_detection.synth import (CAM_H, CAM_W, FPS, HAND_PX, SOURCE_DIR, cut_sprite, detect_landmarks,
                                         detectable, load_backgrounds, paste, read_image, smooth_noise)
from cv.hand_tracking.hand_tracker import STATE_DONE, HandTracker, get_stop_box

BOT_VIDEO_DIR = Path(__file__).resolve().parents[3]   # 레포 바깥 폴더 (main.py의 BOT_VIDEO_DIR과 같은 곳)

OFF_FRAMES = 15         # 사이클 처음 손 없는 구간 (이전 사이클 기록이 끊기게)
FIST_FRAMES = 20        # 주먹 보여 주는 프레임 (앱은 프레임을 건너뛸 수 있어서 synth보다 길게)
MOVE_SECONDS = 2.0      # 이동 시간
DWELL_FRAMES = 75       # 정답 위에서 멈춤 (1초 + 여유)
MIN_DIST = 300.0        # 시작~정답 최소 거리 (px)
MARGIN = np.array([220.0, 160.0])  # 시작점을 화면 끝에서 띄우는 거리 (synth와 같게. 가까우면 팔이 잘려 주먹을 못 찾음)
MAX_TRIES = 80          # 배경·손 사진 조합 대부분은 미리 확인에서 걸러져서 넉넉히


def answer_center(seed: int, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """앱과 같은 생성기·검수기로 seed 문제를 만든다. (문제 이미지, 힌트 이미지 BGR, 정답 중심, 정답 박스)."""
    from cv.problem_gen import ProblemGenerator
    from cv.verification.difficulty_check import DifficultyCheck

    gen = ProblemGenerator(validator=DifficultyCheck())
    pid, image, target = gen.create(seed=seed, verification_method=method)
    box = gen._answers[pid].box                      # 시연 준비용으로만 정답을 꺼낸다 (실제 공격에선 AI가 찾는 부분)
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return image, cv2.cvtColor(target, cv2.COLOR_BGRA2BGR), center, box


def fist_shift(fist, point, target) -> np.ndarray:
    """주먹 → 검지로 바뀔 때 손목이 제자리에 있도록 주먹을 옮길 거리.
    둘 다 검지 끝 기준으로 붙이면 사진이 바뀌는 순간 손 전체가 점프해서 손 인식이 한 프레임 놓치고,
    그러면 시작 준비가 풀려 기록이 시작되지 않는다 (재료 사진에 따라 발생)."""
    fw, pw = detect_landmarks(fist.image)[0], detect_landmarks(point.image)[0]     # 0 = 손목
    return (pw - point.anchor) * (target / point.hand) - (fw - fist.anchor) * (target / fist.hand)


def make_cycle(goal, bg, fist, point, target, rng, level) -> list[np.ndarray]:
    """한 사이클 프레임 (앱이 보는 방향, 반전 전). goal = 문제 이미지 좌표 = 카메라 좌표 (둘 다 1280×720)."""
    shift = fist_shift(fist, point, target)
    lo, hi = MARGIN, np.array([CAM_W, CAM_H]) - MARGIN
    for _ in range(1000):
        start = rng.uniform(lo, hi)
        if np.linalg.norm(goal - start) >= MIN_DIST:
            break
    n = int(MOVE_SECONDS * FPS)
    d = goal - start
    normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    c1 = start + d / 3 + normal * rng.uniform(-0.3, 0.3) * np.linalg.norm(d)
    c2 = start + 2 * d / 3 + normal * rng.uniform(-0.3, 0.3) * np.linalg.norm(d)
    u = min_jerk(np.linspace(0, 1, n))[:, None]
    move = (1 - u)**3 * start + 3 * (1 - u)**2 * u * c1 + 3 * (1 - u) * u**2 * c2 + u**3 * goal
    path = np.vstack([np.repeat(start[None], FIST_FRAMES, axis=0), move, np.repeat(goal[None], DWELL_FRAMES, axis=0)])

    m = len(path)
    if level in ("S2", "S3", "S4"):
        path = path + smooth_noise(m, 2.0, rng)
    angles = smooth_noise(m, 2.0, rng, corr=4.0)[:, 0] if level == "S4" else np.zeros(m)
    scales = 1 + smooth_noise(m, 0.02, rng, corr=4.0)[:, 1] if level == "S4" else np.ones(m)

    frames = [bg.copy() for _ in range(OFF_FRAMES)]
    for i in range(m):
        if i < FIST_FRAMES:
            f = paste(bg, fist, path[i] + shift, 0.0, target / fist.hand)
        else:
            f = paste(bg, point, path[i], angles[i], target / point.hand * scales[i])
        if level in ("S3", "S4"):
            f = np.clip(f + rng.normal(0, 2.0, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def simulate(frames, image, hint, seed_t: int = 0):
    """앱 worker와 같은 흐름: 손 인식 → done이면 멈춤 박스. (HandResult 또는 None)."""
    rng = np.random.default_rng(seed_t)
    tracker = HandTracker(image, hint)
    t = 0.0
    try:
        for f in frames:
            result = tracker.process_landmarks(tracker._detect(f, t), t, f)
            if result is not None and result.state == STATE_DONE:
                return result
            t += 1 / FPS + rng.normal(0, 0.003)
    finally:
        tracker.close()
    return None


def hit(answer_box, result) -> bool:
    from cv.problem_gen.generator import _hit
    return _hit(answer_box, get_stop_box(result))


def write_video(frames, path: Path) -> None:
    """MJPG 화질 100으로 저장한다. mp4v로 압축하면 주먹이 손으로 안 잡혀 기록이 시작되지 않았음."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (CAM_W, CAM_H))
    if not out.isOpened():
        raise RuntimeError(f"영상 파일을 만들 수 없습니다: {path} (영어 경로인지 확인)")
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for f in frames:
        out.write(cv2.flip(f, 1))                    # 앱 Camera가 다시 반전한다
    out.release()


def read_video(path: Path) -> list[np.ndarray]:
    """앱 Camera처럼 읽고 반전한다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"저장한 영상을 열 수 없습니다: {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.flip(f, 1))
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="시드 문제의 정답으로 가는 봇 영상 만들기")
    ap.add_argument("--seed", type=int, default=1234, help="문제 시드 (main.py --bot과 같게)")
    ap.add_argument("--method", default="orb/sift", help="검수기 (앱에서 고른 것과 같게, 기본 orb/sift)")
    ap.add_argument("--level", default="S4", choices=("S1", "S2", "S3", "S4"), help="봇 종류 (기본 S4: 회전·확대·노이즈)")
    ap.add_argument("--out", default=None, help="출력 영상 (기본: 레포 바깥 폴더/bot_<seed>.avi)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else BOT_VIDEO_DIR / f"bot_{args.seed}.avi"

    print(f"[1/3] seed {args.seed} 문제 생성 ({args.method})...", flush=True)
    image, hint, goal, box = answer_center(args.seed, args.method)
    print(f"      정답 박스 {box}, 중심 ({goal[0]:.0f}, {goal[1]:.0f})")

    print("[2/3] 봇 재료 준비...", flush=True)
    rng = np.random.default_rng(args.seed)
    bgs = load_backgrounds()
    fist = cut_sprite(read_image(SOURCE_DIR / "주먹.jpg"))
    points = [cut_sprite(read_image(Path(f))) for f in sorted(glob.glob(str(SOURCE_DIR / "검지*.jpg")))]

    # 시도마다 저장 → 다시 읽기(앱 Camera와 같은 반전)까지 거친 프레임으로 확인한다 (압축 뒤에도 잡히는지)
    print("[3/3] 저장한 영상을 앱처럼 읽어 손 인식 → 정답 → 진위 검출 확인...", flush=True)
    verdict = best = None
    for attempt in range(1, MAX_TRIES + 1):
        bg = bgs[int(rng.integers(len(bgs)))]
        point = points[int(rng.integers(len(points)))]
        target = rng.uniform(*HAND_PX)
        if not detectable(bg, fist, point, target):
            continue
        frames = make_cycle(goal, bg, fist, point, target, rng, args.level)
        write_video(frames, out)
        result = simulate(read_video(out), image, hint)
        if result is None:
            print(f"      시도 {attempt}: 손 인식 done 안 됨")
            continue
        if not hit(box, result):
            print(f"      시도 {attempt}: 멈춘 위치가 정답과 안 맞음")
            continue
        verdict, best = detector.check(result), frames
        print(f"      시도 {attempt}: 정답 O, 진위 검출 {verdict[0]} | {verdict[1]}")
        if verdict[0] is False:
            break
    if verdict is None:
        raise SystemExit("봇 영상을 못 만들었습니다. 다른 --seed로 해 보세요.")
    ok = verdict[0] is False
    if not ok:
        write_video(best, out)                       # 마지막 시도가 실패였을 수 있어서 정답 맞힌 영상으로 다시 저장
    print(f"\n{'봇 판정 OK' if ok else '주의: 봇으로 안 잡힘 (다른 --seed나 --level로 다시)'} → {out}")
    print(f"시연: python main.py --bot {args.seed}" + (f' --camera "{out}"' if args.out else ""))


if __name__ == "__main__":
    main()
