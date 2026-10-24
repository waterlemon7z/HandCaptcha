"""손 인식 데모: 문제 이미지 + 손끝 커서 + dwell 게이지.

실행:
    python demo_hand.py              # 웹캠 (모델 파일 필요)
    python demo_hand.py --mouse      # 웹캠 없이 마우스로 손끝 흉내

키: n = 다음 문제, r = 현재 문제 다시, q/ESC = 종료
(OpenCV putText는 한글을 못 그려서 화면 문구는 영어)
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from hand_tracker import (
    STATUS_DWELLING,
    STATUS_NO_HAND,
    STATUS_SELECTED,
    STATUS_TRACKING,
    HandState,
    HandTracker,
)
from mock_captcha import MockCaptcha

WINDOW = "HandCAPTCHA - hand tracking demo"
GAP = 20             # 좌우 이미지 사이 간격 (px)
BAR_H = 70           # 아래 정보 표시줄 높이
AUTO_NEXT_SEC = 3.0  # 결과 표시 후 다음 문제로 넘어가는 시간

STATUS_TEXT = {
    STATUS_NO_HAND: "No hand - show your index finger",
    STATUS_TRACKING: "Tracking",
    STATUS_DWELLING: "Hold still to select...",
    STATUS_SELECTED: "Selected",
}


def compose_canvas(left: np.ndarray, right: np.ndarray, gap: int) -> np.ndarray:
    """왼쪽 | gap | 오른쪽 으로 붙인 문제 캔버스."""
    h = left.shape[0]
    spacer = np.full((h, gap, 3), 40, np.uint8)
    return np.hstack([left, spacer, right])


def draw_frame(
    canvas: np.ndarray,
    state: HandState,
    dwell_radius: float,
    result: Optional[dict],
) -> np.ndarray:
    """캔버스 위에 커서, 게이지, 상태/결과 표시줄을 그린 화면을 만든다."""
    h, w = canvas.shape[:2]
    out = np.zeros((h + BAR_H, w, 3), np.uint8)
    out[:h] = canvas

    if state.point is not None:
        cx, cy = int(round(state.point[0])), int(round(state.point[1]))
        color = (0, 200, 0) if state.selected else (0, 140, 255)
        # dwell 반경 (얇은 원)
        cv2.circle(out, (cx, cy), int(dwell_radius), (255, 255, 255), 1, cv2.LINE_AA)
        # dwell 게이지 (12시 방향부터 시계 방향으로 차오름)
        if state.progress > 0:
            cv2.ellipse(out, (cx, cy), (22, 22), -90, 0, 360 * state.progress, color, 4, cv2.LINE_AA)
        # 손끝 커서
        cv2.circle(out, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)

    y1, y2 = h + 28, h + 56
    status = STATUS_TEXT.get(state.status, state.status)
    cv2.putText(out, f"{status}  ({state.progress * 100:3.0f}%)", (10, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

    if result is not None:
        ok = result["ok"]
        label = "CORRECT" if ok else "WRONG"
        cv2.putText(out, label, (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 220, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
        info = (f"answer={result['side']} {result['answer']}  "
                f"trajectory={result['n']} pts / {result['duration']:.2f}s   [n] next")
        cv2.putText(out, info, (140, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "Point at the difference and hold for a moment.  [q] quit",
                    (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="HandCAPTCHA 손 인식 데모")
    ap.add_argument("--camera", type=int, default=0, help="웹캠 번호")
    ap.add_argument("--mouse", action="store_true", help="웹캠 대신 마우스 좌표 사용")
    ap.add_argument("--smoothing", type=float, default=0.4)
    ap.add_argument("--dwell-radius", type=float, default=30)
    ap.add_argument("--dwell-time", type=float, default=1.0)
    args = ap.parse_args()

    captcha = MockCaptcha(size=(480, 360))
    img_hash, left, right = captcha.create()
    canvas = compose_canvas(left, right, GAP)

    overrides = dict(smoothing=args.smoothing, dwell_radius=args.dwell_radius,
                     dwell_time=args.dwell_time)
    if args.mouse:
        overrides["active_margin"] = 0.0  # 마우스는 캔버스 좌표를 그대로 사용
    tracker = HandTracker(captcha.size, gap=GAP, **overrides)
    cw, ch = tracker.canvas_size

    cv2.namedWindow(WINDOW)
    mouse: list[Optional[tuple[int, int]]] = [None]
    cap = None
    if args.mouse:
        def on_mouse(event, x, y, flags, param):
            # 정보 표시줄 위로 내려가면 '손이 사라진 것'으로 취급
            mouse[0] = (x, y) if y < ch else None
        cv2.setMouseCallback(WINDOW, on_mouse)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"웹캠 {args.camera}번을 열 수 없습니다. --mouse 로 실행해 보세요.")

    result: Optional[dict] = None
    result_time = 0.0
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("웹캠 프레임을 읽지 못했습니다.")
                    break
                state = tracker.update(frame)
                # 작은 거울 미리보기 (손 위치 확인용)
                preview = cv2.flip(cv2.resize(frame, (320, 240)), 1)
                if state.point is not None:
                    cv2.putText(preview, "hand", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                cv2.imshow("webcam", preview)
            else:
                m = mouse[0]
                norm = None if m is None else (m[0] / (cw - 1), m[1] / (ch - 1))
                state = tracker.update_point(norm)

            # 선택이 확정된 첫 프레임에 정답 확인 + 궤적 전달
            if state.selected and result is None:
                trajectory = tracker.get_trajectory()
                # TODO(통합): trajectory를 진위 검출 모듈에 넘긴다
                result = {
                    "ok": captcha.solve(img_hash, state.answer),
                    "side": state.side,
                    "answer": state.answer,
                    "n": len(trajectory),
                    "duration": trajectory[-1][2] - trajectory[0][2] if trajectory else 0.0,
                }
                result_time = time.monotonic()
                print(f"선택: {state.side} {state.answer} -> "
                      f"{'정답' if result['ok'] else '오답'}, 궤적 {result['n']}점")

            cv2.imshow(WINDOW, draw_frame(canvas, state, tracker.config.dwell_radius, result))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            next_problem = key == ord("n") or (
                result is not None and time.monotonic() - result_time > AUTO_NEXT_SEC)
            if next_problem:
                img_hash, left, right = captcha.create()
                canvas = compose_canvas(left, right, GAP)
                tracker.reset()
                result = None
            elif key == ord("r"):
                tracker.reset()
                result = None
    finally:
        if cap is not None:
            cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
