"""손인식 데모.

실행: python demo.py
- 주먹을 쥐면 기록 시작 → 손을 움직임 → 1초 멈추면 완료(멈춘 위치에 초록 표시)
- 이동 경로는 화면에 그리지 않고, 완료되면 recordings/ 에 이동 경로와 멈춘 위치를 각각 CSV로 저장
- 화면 아래 '다시하기' 버튼 클릭 또는 r: 리셋, q: 종료
샘플 이미지(samples/problem.png, samples/hint.png)가 없으면 코드로 임시 생성한다.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from hand_tracker import STATE_DONE, HandTracker, _draw_label

SAMPLES = Path(__file__).resolve().parent / "samples"
RECORDINGS = Path(__file__).resolve().parent / "recordings"   # 완료된 이동 경로 CSV 저장 위치
WINDOW = "HandTracker demo"
PREVIEW_W, PREVIEW_H = 256, 144   # 카메라 미리보기 크기 (문제 이미지 왼쪽 아래 구석)
BAR_H = 60                        # 화면 아래 버튼 줄 높이
BUTTON = (20, 10, 220, 50)        # 버튼 줄 안의 '다시하기' 버튼 (x1, y1, x2, y2)


def make_sample_images():
    """임시 문제 이미지(1280×720)와 힌트 이미지(찾을 물체만 잘라 낸 것)."""
    rng = np.random.default_rng(7)
    problem = np.full((720, 1280, 3), 225, np.uint8)
    for _ in range(40):
        color = tuple(int(c) for c in rng.integers(60, 200, 3))
        x, y, s = int(rng.integers(0, 1280)), int(rng.integers(0, 720)), int(rng.integers(20, 70))
        if rng.random() < 0.5:
            cv2.rectangle(problem, (x - s, y - s), (x + s, y + s), color, -1)
        else:
            cv2.circle(problem, (x, y), s, color, -1, cv2.LINE_AA)

    # 찾을 대상: 노란 별
    cx, cy, r = 880, 260, 60
    angles = np.deg2rad(np.arange(10) * 36 - 90)
    radii = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
    star = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1).astype(np.int32)
    cv2.fillPoly(problem, [star], (0, 220, 255), cv2.LINE_AA)
    cv2.polylines(problem, [star], True, (0, 0, 0), 2, cv2.LINE_AA)

    hint = problem[cy - r - 10:cy + r + 10, cx - r - 10:cx + r + 10].copy()
    return problem, hint


def imread(path: Path):
    """cv2.imread는 Windows에서 한글 경로를 못 읽어서, 바이트로 읽은 뒤 디코딩한다. 없으면 None."""
    if not path.exists():
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_images():
    problem = imread(SAMPLES / "problem.png")
    hint = imread(SAMPLES / "hint.png")
    if problem is None or hint is None:
        print("샘플 이미지가 없어 임시 이미지를 만듭니다.")
        return make_sample_images()
    if problem.shape[:2] != (720, 1280):
        print(f"문제 이미지를 1280×720으로 맞춥니다 (원래 {problem.shape[1]}×{problem.shape[0]}).")
        problem = cv2.resize(problem, (1280, 720))
    return problem, hint


def save_result(result) -> tuple[Path, Path]:
    """이동 경로와 마지막 멈춘 위치를 각각 다른 CSV로 저장한다.

    - path_날짜_시각.csv: 기록 시작부터 완료까지 움직인 경로 전체 (t, x, y)
    - stop_날짜_시각.csv: 마지막에 멈춘 위치 한 점 (x, y)
    """
    RECORDINGS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path_file = RECORDINGS / f"path_{stamp}.csv"
    stop_file = RECORDINGS / f"stop_{stamp}.csv"
    data = np.column_stack([result.timestamps, result.trajectory])
    np.savetxt(path_file, data, delimiter=",", fmt=["%.3f", "%d", "%d"], header="t,x,y", comments="")
    np.savetxt(stop_file, [result.stop_position], delimiter=",", fmt="%d", header="x,y", comments="")
    return path_file, stop_file


def in_button(x: int, y: int) -> bool:
    """버튼 줄 기준 좌표 (x, y)가 '다시하기' 버튼 안인가."""
    x1, y1, x2, y2 = BUTTON
    return x1 <= x <= x2 and y1 <= y <= y2


def make_button_bar(width: int, done: bool) -> np.ndarray:
    """화면 아래 버튼 줄. 완료 상태면 버튼을 초록색으로 강조하고 안내 문구를 바꾼다."""
    bar = np.full((BAR_H, width, 3), 30, np.uint8)
    x1, y1, x2, y2 = BUTTON
    cv2.rectangle(bar, (x1, y1), (x2, y2), (0, 170, 0) if done else (90, 90, 90), -1)
    cv2.rectangle(bar, (x1, y1), (x2, y2), (255, 255, 255), 1)
    _draw_label(bar, "다시하기 (R)", "RETRY (R)", ((x1 + x2) // 2, (y1 + y2) // 2))
    msg = ("완료! 다시 하려면 버튼을 누르세요" if done
           else "주먹을 쥐면 기록 시작, 1초 멈추면 완료")
    _draw_label(bar, msg, "Done! Click RETRY" if done else "Make a fist to start",
                (x2 + 30, (y1 + y2) // 2), align_left=True)
    return bar


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 웹캠이 연결돼 있는지, 다른 프로그램(줌 등)이 쓰고 있지 않은지 확인하세요.")
        return

    problem, hint = load_images()
    tracker = HandTracker(problem, hint)
    reported = False
    bar_width = tracker.width + tracker.panel_width
    bars = {done: make_button_bar(bar_width, done) for done in (False, True)}

    clicked = [False]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and in_button(x, y - tracker.height):
            clicked[0] = True

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다.")
                break

            result = tracker.process(frame)
            canvas = tracker.render(result)

            # 카메라 미리보기: 문제 이미지 왼쪽 아래 구석 (힌트 패널과 겹치지 않음)
            preview = cv2.resize(cv2.flip(frame, 1) if tracker.mirror else frame, (PREVIEW_W, PREVIEW_H))
            canvas[720 - PREVIEW_H:720, 0:PREVIEW_W] = preview
            cv2.rectangle(canvas, (0, 720 - PREVIEW_H), (PREVIEW_W - 1, 719), (255, 255, 255), 1)

            if result.state == STATE_DONE and not reported:
                path_file, stop_file = save_result(result)
                print(f"완료: 멈춘 위치={result.stop_position} → {stop_file.name}")
                print(f"      이동 경로 {len(result.trajectory)}점, {result.timestamps[-1]:.2f}초 "
                      f"→ {path_file.name}")
                reported = True

            # 버튼 줄은 render 화면 아래에 붙인다 (힌트 패널과 겹치지 않음)
            cv2.imshow(WINDOW, np.vstack([canvas, bars[result.state == STATE_DONE]]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") or clicked[0]:
                tracker.reset()
                reported = False
                clicked[0] = False
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
