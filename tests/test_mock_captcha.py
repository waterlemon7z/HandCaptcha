"""MockCaptcha와 HandTracker 연결 테스트."""
import numpy as np

from hand_tracker import HandTracker
from mock_captcha import MockCaptcha

BOX = (300, 90, 380, 170)


def test_create_returns_same_size_images_with_difference_in_box():
    cap = MockCaptcha(size=(480, 360), answer_box=BOX, seed=0)
    img_hash, left, right = cap.create()
    assert isinstance(img_hash, str)
    assert left.shape == right.shape == (360, 480, 3)
    diff = np.any(left != right, axis=2)
    ys, xs = np.nonzero(diff)
    assert len(xs) > 0
    assert xs.min() >= BOX[0] and xs.max() <= BOX[2]
    assert ys.min() >= BOX[1] and ys.max() <= BOX[3]


def test_solve_correct_and_wrong():
    cap = MockCaptcha(answer_box=BOX, seed=0)
    h1, _, _ = cap.create()
    assert cap.solve(h1, (340, 130))
    h2, _, _ = cap.create()
    assert not cap.solve(h2, (10, 10))


def test_solve_only_once_and_unknown_hash():
    cap = MockCaptcha(answer_box=BOX, seed=0)
    h, _, _ = cap.create()
    assert cap.solve(h, (340, 130))
    assert not cap.solve(h, (340, 130))
    assert not cap.solve("nope", (340, 130))


def test_tracker_answer_solves_mock_captcha():
    """오른쪽 이미지의 정답 위치를 1초 이상 짚으면 solve가 True."""
    cap = MockCaptcha(size=(480, 360), answer_box=BOX, seed=1)
    img_hash, _, _ = cap.create()
    tr = HandTracker((480, 360), gap=20, smoothing=0.4, active_margin=0.0)
    cw, ch = tr.canvas_size
    target = (480 + 20 + 340, 130)  # 캔버스 기준 오른쪽 이미지의 (340, 130)

    t, state = 0.0, None
    for i in range(60):
        jitter = 3 if i % 2 else -3
        state = tr.update_point(((target[0] + jitter) / (cw - 1), target[1] / (ch - 1)), t)
        t += 1 / 30
        if state.selected:
            break
    assert state.selected and state.side == "right"
    assert cap.solve(img_hash, state.answer)
    assert len(tr.get_trajectory()) >= 30
