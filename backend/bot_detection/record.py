"""사람 손 궤적 녹화 (진위검출 기준 데이터 수집용).

손인식(HandTracker)을 웹캠으로 돌리다가 done이 되면 HandResult를 .npz로 저장한다.
카메라 조건은 앱(cv/camera.py)과 같게: 1280×720 요청, 좌우 반전 후 HandTracker에 넣음.
동작: 주먹 → 검지 펴기(기록 시작) → 이동 → 1초 멈춤(done, 저장). 저장 후 잠깐 뒤 자동 reset되고,
다음 녹화는 다시 주먹부터 (HandTracker 기본 start_gesture="point").
거의 움직이지 않은 기록은 저장하지 않는다.

실행 (레포 루트에서)
    python cv/hand_tracking/download_model.py      # 처음 한 번 (모델 파일)
    python -m backend.bot_detection.record
저장 위치: backend/bot_detection/test/data/human/ (gitignore, 웹캠 영상 포함)

키: q = 종료, r = 지금 기록 버리고 다시
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cv.hand_tracking.hand_tracker import DEFAULT_MODEL, STATE_DONE, HandTracker

OUT_DIR = Path(__file__).resolve().parent / "test" / "data" / "human"
CAM_W, CAM_H = 1280, 720    # 앱 Camera와 같은 요청 해상도
RESET_DELAY = 1.0       # done 후 이만큼 지나면 자동 reset (초)
MIN_SPAN = 30.0         # 원본 궤적의 이동 범위(가로·세로 중 큰 값)가 이 픽셀 미만이면 저장 안 함
MAX_READ_FAILS = 60     # 카메라 프레임을 이만큼 연속 못 읽으면 종료 (가끔 놓치는 건 무시)


def dummy_images(width: int = 1280, height: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """녹화용 가짜 문제·힌트 이미지. 위치 감 잡기 쉽게 격자만 그린다."""
    problem = np.full((height, width, 3), 90, np.uint8)
    for x in range(0, width, 160):
        cv2.line(problem, (x, 0), (x, height - 1), (130, 130, 130), 1)
    for y in range(0, height, 160):
        cv2.line(problem, (0, y), (width - 1, y), (130, 130, 130), 1)
    hint = np.full((120, 120, 3), 160, np.uint8)
    return problem, hint


def save_result(result, out_dir: Path = OUT_DIR, name: str | None = None) -> Path:
    """done 상태의 HandResult를 .npz로 저장한다. name이 없으면 human_날짜_시각."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name or f'human_{datetime.now():%Y%m%d_%H%M%S}'}.npz"
    if result.frames:
        # 손끝이 화면 밖 멀리 가면 손인식 crop 크기가 깨짐 → 그 프레임은 검정으로 (영상 검사에서 제외됨)
        shapes = [f.shape for f in result.frames]
        common = max(set(shapes), key=shapes.count)
        bad = sum(s != common for s in shapes)
        if bad:
            print(f"  (crop 크기가 깨진 프레임 {bad}장은 검정으로 저장)")
        frames = np.stack([f if f.shape == common else np.zeros(common, np.uint8) for f in result.frames])
    else:
        frames = np.empty((0, 0, 0, 3), np.uint8)
    np.savez_compressed(
        path,
        trajectory=result.trajectory,
        raw_trajectory=result.raw_trajectory,
        timestamps=result.timestamps,
        frames=frames,
        frame_timestamps=result.frame_timestamps if result.frame_timestamps is not None else np.empty(0),
        frame_size=np.asarray(result.frame_size if result.frame_size else (0, 0)),
    )
    return path


def load_recording(path: str | Path) -> dict[str, np.ndarray]:
    """save_result로 저장한 .npz를 dict로 읽는다."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def main(camera: int = 0) -> None:
    if not Path(DEFAULT_MODEL).exists():
        raise SystemExit(f"모델 파일이 없음: {DEFAULT_MODEL}\n먼저 실행: python cv/hand_tracking/download_model.py")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {camera}번을 열 수 없습니다")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    real_w, real_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 해상도: {real_w}×{real_h}" + ("" if (real_w, real_h) == (CAM_W, CAM_H)
                                               else f"  (요청 {CAM_W}×{CAM_H}을 지원 안 함, 앱도 같은 해상도로 돌아감)"))
    tracker = HandTracker(*dummy_images())
    cv2.namedWindow("record", cv2.WINDOW_NORMAL)
    saved, done_at, fails = 0, None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1                  # 웹캠이 가끔 프레임을 놓침 → 바로 끄지 않고 다시 읽음
                if fails >= MAX_READ_FAILS:
                    print(f"카메라 프레임을 {MAX_READ_FAILS}번 연속 못 읽어서 종료 (다른 프로그램이 카메라를 쓰는지 확인)")
                    break
                time.sleep(0.01)
                continue
            fails = 0
            frame = cv2.flip(frame, 1)      # 앱 Camera와 같이 좌우 반전 (거울처럼)
            result = tracker.process(frame)
            if result is None:
                continue

            if result.state == STATE_DONE and done_at is None:
                done_at = time.monotonic()
                span = float(np.ptp(result.raw_trajectory, axis=0).max())
                if span < MIN_SPAN:
                    print(f"건너뜀: 이동 범위 {span:.0f}px (< {MIN_SPAN:.0f}px)")
                else:
                    path = save_result(result)
                    saved += 1
                    print(f"[{saved}] {path.name}  점 {len(result.timestamps)}개, "
                          f"{result.timestamps[-1]:.2f}초, 이동 범위 {span:.0f}px")
            if done_at is not None and time.monotonic() - done_at >= RESET_DELAY:
                tracker.reset()             # 다음 기록은 다시 주먹 → 검지 펴기부터
                done_at = None

            view = tracker.render(result)
            status = f"saved: {saved}   fist -> point to start"
            cv2.putText(view, status, (10, view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("record", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset()
                done_at = None
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    print(f"총 {saved}개 저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
