"""사람 손 궤적 녹화 (진위검출 기준 데이터 수집용).

손인식(HandTracker)을 웹캠으로 돌리다가 done이 되면 HandResult를 .npz로 저장한다.
저장 후 잠깐 뒤 자동으로 reset되고, 손을 한 번 펴면 다음 녹화를 받는다.
(주먹을 쥔 채로 reset되면 바로 기록이 시작돼 1초 뒤 "멈춤"으로 끝나 버리므로)
거의 움직이지 않은 기록은 저장하지 않는다.

실행 (레포 루트에서, hand_tracker.py와 models/hand_landmarker.task 필요)
    python -m cv.bot_detection.record

키: q = 종료, r = 지금 기록 버리고 다시
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hand_tracker import HAND_FIST, STATE_DONE, HandTracker

OUT_DIR = Path(__file__).resolve().parents[1] / "test" / "data" / "human"
RESET_DELAY = 1.0       # done 후 이만큼 지나면 자동 reset (초)
MIN_SPAN = 30.0         # 원본 궤적의 이동 범위(가로·세로 중 큰 값)가 이 픽셀 미만이면 저장 안 함


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
    frames = np.stack(result.frames) if result.frames else np.empty((0, 0, 0, 3), np.uint8)
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
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {camera}번을 열 수 없습니다")
    tracker = HandTracker(*dummy_images())
    cv2.namedWindow("record", cv2.WINDOW_NORMAL)
    saved, done_at = 0, None
    armed = True            # False면 손을 펼 때까지 기록을 받지 않음
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = tracker.process(frame)

            if not armed:
                if result.hand_status == HAND_FIST:
                    tracker.reset()     # 주먹 쥔 채로 시작된 기록은 버린다
                else:
                    armed = True

            if armed and result.state == STATE_DONE and done_at is None:
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
                tracker.reset()
                done_at = None
                armed = False

            view = tracker.render(result)
            status = f"saved: {saved}" + ("" if armed else "   open your hand")
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
