"""카메라 없이 가짜 랜드마크/좌표로 손인식 로직을 검증한다."""
import numpy as np
import pytest

from hand_tracker import (
    HAND_FIST,
    HAND_NONE,
    HAND_OPEN,
    STATE_DONE,
    STATE_IDLE,
    STATE_RECORDING,
    HandTracker,
    classify_hand,
    count_folded_fingers,
    palm_center,
)

W, H = 1280, 720
DT = 1 / 30


def make_images():
    rng = np.random.default_rng(0)
    problem = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    hint = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    return problem, hint


def make_tracker(**kw):
    kw.setdefault("alpha", 1.0)  # 기본은 보정 없이 (좌표를 그대로 비교하려고)
    return HandTracker(*make_images(), **kw)


def make_landmarks(cx=0.5, cy=0.5, folded=0):
    """손바닥 중심이 (cx, cy)이고 검지부터 folded개 손가락이 접힌 가짜 랜드마크 21개."""
    pts = [(cx, cy)] * 21
    pts = [list(p) for p in pts]
    pts[0] = [cx, cy + 0.2]                       # 손목
    mcp_x = [cx - 0.06, cx - 0.02, cx + 0.02, cx + 0.06]
    for i, (mcp, pip, tip) in enumerate([(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]):
        pts[mcp] = [mcp_x[i], cy - 0.05]          # 손가락 뿌리 → 손바닥 중심 y가 정확히 cy
        pts[pip] = [mcp_x[i], cy - 0.10]
        pts[pip + 1] = [mcp_x[i], cy - 0.13]      # DIP
        # 펴진 손가락: 끝이 PIP보다 멀리 / 접힌 손가락: 끝이 손바닥 쪽으로
        pts[tip] = [mcp_x[i], cy + 0.02] if i < folded else [mcp_x[i], cy - 0.16]
    return [tuple(p) for p in pts]


def feed(tracker, points, status=HAND_FIST, t0=0.0, dt=DT):
    """문제 이미지 좌표 목록을 차례로 넣는다. None은 손 없음."""
    result, t = None, t0
    for p in points:
        result = tracker.update(p, HAND_NONE if p is None else status, t)
        t += dt
    return result, t


# ---------------------------------------------------------------- 랜드마크 → (위치, 상태)

def test_palm_center_is_mean_of_palm_points():
    assert palm_center(make_landmarks(0.3, 0.6)) == pytest.approx((0.3, 0.6))


@pytest.mark.parametrize("folded, expected", [(0, HAND_OPEN), (2, HAND_OPEN),
                                              (3, HAND_FIST), (4, HAND_FIST)])
def test_fist_needs_three_folded_fingers(folded, expected):
    lm = make_landmarks(folded=folded)
    assert count_folded_fingers(lm) == folded
    assert classify_hand(lm) == expected


def test_min_folded_is_configurable():
    assert classify_hand(make_landmarks(folded=2), min_folded=2) == HAND_FIST


def test_landmarks_map_to_problem_image_with_mirror():
    tr = make_tracker()
    r = tr.process_landmarks(make_landmarks(0.25, 0.5), t=0.0)
    # 좌우 반전: 카메라 x=0.25 → 화면 x=0.75
    assert r.position == (round(0.75 * (W - 1)), round(0.5 * (H - 1)))
    assert r.hand_status == HAND_OPEN


def test_landmarks_without_mirror():
    tr = make_tracker(mirror=False)
    r = tr.process_landmarks(make_landmarks(0.25, 0.5), t=0.0)
    assert r.position[0] == round(0.25 * (W - 1))


def test_no_landmarks_means_no_hand():
    tr = make_tracker()
    r = tr.process_landmarks(None, t=0.0)
    assert r.position is None
    assert r.hand_status == HAND_NONE
    assert r.state == STATE_IDLE


def test_fist_landmarks_start_recording():
    tr = make_tracker()
    r = tr.process_landmarks(make_landmarks(folded=4), t=0.0)
    assert r.state == STATE_RECORDING
    assert len(r.trajectory) == 1


# ---------------------------------------------------------------- 상태 전환

def test_open_hand_stays_idle():
    tr = make_tracker()
    r, _ = feed(tr, [(100 + i * 10, 100) for i in range(30)], status=HAND_OPEN)
    assert r.state == STATE_IDLE
    assert len(r.trajectory) == 0
    assert r.position == (390, 100)


def test_idle_recording_done():
    tr = make_tracker()
    move = [(100 + i * 10, 200 + i * 5) for i in range(30)]   # 0~0.97초: 이동
    still = [(390, 345)] * 32                                   # 1초 이상 정지
    r, t = feed(tr, move[:1])
    assert r.state == STATE_RECORDING
    r, t = feed(tr, move[1:], t0=t)
    assert r.state == STATE_RECORDING
    r, t = feed(tr, still, t0=t)
    assert r.state == STATE_DONE
    assert r.box == (100, 200, 390, 345)
    assert r.stop_position == (390, 345)
    assert r.trajectory.shape == (len(r.timestamps), 2)
    assert r.trajectory[0].tolist() == [100, 200]


def test_recording_continues_with_open_hand():
    """기록 시작 후에는 손을 펴도 계속 기록한다 (시작만 주먹)."""
    tr = make_tracker()
    feed(tr, [(100, 100)], status=HAND_FIST)
    r, _ = feed(tr, [(100 + i * 20, 100) for i in range(1, 20)], status=HAND_OPEN, t0=DT)
    assert r.state == STATE_RECORDING
    assert len(r.trajectory) == 20


def test_stop_is_time_based_not_frame_based():
    # 프레임이 많아도 1초가 안 되면 done이 아님
    tr = make_tracker()
    r, _ = feed(tr, [(500, 300)] * 100, dt=0.009)   # 0.89초
    assert r.state == STATE_RECORDING
    # 프레임이 적어도 1초가 지나면 done
    tr = make_tracker()
    r, _ = feed(tr, [(500, 300)] * 6, dt=0.2)       # 0, 0.2, ..., 1.0초
    assert r.state == STATE_DONE


def test_stop_uses_only_last_second():
    tr = make_tracker()
    feed(tr, [(100 + i * 30, 300) for i in range(20)])            # 이동 (0~0.63초)
    r, t = feed(tr, [(800, 300)] * 20, t0=20 * DT)                # 0.63초 정지 → 아직
    assert r.state == STATE_RECORDING
    r, _ = feed(tr, [(800, 300)] * 15, t0=t)                       # 마지막 1초 동안 정지
    assert r.state == STATE_DONE


@pytest.mark.parametrize("jitter, expected", [(14, STATE_DONE), (15, STATE_RECORDING)])
def test_stop_threshold_15_pixels(jitter, expected):
    tr = make_tracker()
    pts = [(500 + (jitter if i % 2 else 0), 300) for i in range(40)]
    r, _ = feed(tr, pts)
    assert r.state == expected


def test_timestamps_are_seconds_from_recording_start():
    tr = make_tracker()
    feed(tr, [None] * 5, t0=100.0)
    r, _ = feed(tr, [(200, 200), (210, 200), (220, 200)], t0=100.5, dt=0.1)
    assert r.timestamps == pytest.approx([0.0, 0.1, 0.2])


def test_done_holds_until_reset():
    tr = make_tracker()
    r, t = feed(tr, [(300, 300)] * 40)
    assert r.state == STATE_DONE
    box, n = r.box, len(r.trajectory)
    r, _ = feed(tr, [(900, 600)] * 10, status=HAND_FIST, t0=t)
    assert r.state == STATE_DONE and r.box == box and len(r.trajectory) == n
    assert r.position == (900, 600)   # 커서는 계속 움직임

    tr.reset()
    r = tr.update((900, 600), HAND_OPEN, t + 1)
    assert r.state == STATE_IDLE and r.box is None and len(r.trajectory) == 0


# ---------------------------------------------------------------- 손 놓침

def test_short_loss_keeps_recording():
    tr = make_tracker()
    r, t = feed(tr, [(100 + i * 10, 300) for i in range(10)])
    r, t = feed(tr, [None] * 12, t0=t)          # 마지막 점 이후 0.4초
    assert r.state == STATE_RECORDING
    assert r.position is None and r.hand_status == HAND_NONE
    r, _ = feed(tr, [(300, 300)], status=HAND_OPEN, t0=t)
    assert r.state == STATE_RECORDING
    assert len(r.trajectory) == 11


def test_loss_over_half_second_cancels():
    tr = make_tracker()
    r, t = feed(tr, [(100 + i * 10, 300) for i in range(10)])
    last_t = t - DT
    r = tr.update(None, HAND_NONE, last_t + 0.5)
    assert r.state == STATE_IDLE
    assert len(r.trajectory) == 0 and len(r.timestamps) == 0


# ---------------------------------------------------------------- 문제 이미지 밖 / 힌트 패널

def test_hand_on_hint_panel_is_ignored_in_idle():
    tr = make_tracker()
    r, _ = feed(tr, [(W + 100, 300)] * 10, status=HAND_FIST)
    assert r.position is None
    assert r.state == STATE_IDLE
    assert len(r.trajectory) == 0


def test_hand_on_hint_panel_not_added_to_trajectory():
    tr = make_tracker()
    r, t = feed(tr, [(1000, 300), (1100, 300), (1200, 300)])
    r, t = feed(tr, [(W + 50, 300)] * 5, t0=t)       # 0.17초 동안 패널 쪽
    assert r.state == STATE_RECORDING
    assert len(r.trajectory) == 3
    assert r.trajectory[:, 0].max() <= W - 1


def test_hand_on_hint_panel_too_long_cancels():
    tr = make_tracker()
    r, t = feed(tr, [(1000, 300), (1100, 300)])
    r, _ = feed(tr, [(W + 50, 300)] * 20, t0=t)       # 0.67초
    assert r.state == STATE_IDLE


def test_landmarks_outside_camera_frame_are_ignored():
    """카메라 밖으로 걸친 손(정규화 좌표 범위 밖)은 힌트 패널로 매핑되지 않고 무시된다."""
    tr = make_tracker()
    r = tr.process_landmarks(make_landmarks(-0.05, 0.5, folded=4), t=0.0)  # 반전 후 x = 1.05
    assert r.position is None
    assert r.state == STATE_IDLE


def test_box_is_clipped_and_ordered():
    tr = make_tracker()
    tr._points = [(-5.0, 10.0), (W + 20.0, 700.0), (50.0, H + 3.0)]
    x1, y1, x2, y2 = tr._make_box()
    assert (x1, y1, x2, y2) == (0, 10, W - 1, H - 1)
    assert x1 < x2 and y1 < y2


def test_box_never_degenerate():
    tr = make_tracker()
    r, _ = feed(tr, [(500, 300)] * 40)
    x1, y1, x2, y2 = r.box
    assert x1 < x2 and y1 < y2
    tr._points = [(W - 1.0, H - 1.0)]
    x1, y1, x2, y2 = tr._make_box()
    assert x1 < x2 <= W - 1 and y1 < y2 <= H - 1


# ---------------------------------------------------------------- EMA

def test_ema_smoothing():
    tr = make_tracker(alpha=0.5)
    tr.update((100, 100), HAND_OPEN, 0.0)
    r = tr.update((200, 300), HAND_OPEN, DT)
    assert r.position == (150, 200)


def test_ema_restarts_after_hand_lost():
    tr = make_tracker(alpha=0.5)
    tr.update((100, 100), HAND_OPEN, 0.0)
    tr.update(None, HAND_NONE, DT)
    r = tr.update((200, 300), HAND_OPEN, 2 * DT)
    assert r.position == (200, 300)


def test_bad_alpha():
    with pytest.raises(ValueError):
        make_tracker(alpha=0)


# ---------------------------------------------------------------- render

def busy_result(tr):
    r, _ = feed(tr, [(100 + i * 25, 100 + i * 15) for i in range(40)] + [(1100, 700)] * 40)
    assert r.state == STATE_DONE
    return r


def test_render_size_and_layout():
    tr = make_tracker()
    canvas = tr.render(tr.update(None, HAND_NONE, 0.0))
    assert canvas.shape == (H, W + 320, 3)
    assert canvas.dtype == np.uint8


def test_render_hint_panel_untouched_by_hand_overlay():
    tr = make_tracker()
    empty = tr.render(tr.update(None, HAND_NONE, 0.0))
    r = busy_result(tr)
    # 힌트 패널 바로 옆(문제 이미지 오른쪽 끝)까지 손이 가도 패널은 그대로
    r.position = (W - 1, 360)
    busy = tr.render(r)
    assert np.array_equal(busy[:, W:], empty[:, W:])
    assert not np.array_equal(busy[:, :W], empty[:, :W])   # 오버레이는 문제 이미지 쪽에만


def test_render_does_not_draw_trajectory():
    tr = make_tracker()
    r = busy_result(tr)
    assert len(r.trajectory) > 2
    without = tr.render(r)
    r.trajectory = np.empty((0, 2), np.int32)
    assert np.array_equal(tr.render(r), without)


def test_trajectory_kept_from_start_until_done():
    """기록 시작부터 완료 프레임까지의 모든 점이 빠짐없이 남는다."""
    tr = make_tracker()
    move = [(100 + i * 10, 200) for i in range(60)]   # 2초 동안 이동
    still = [(800, 200)] * 31                          # 마지막 이동 점과 떨어진 곳에서 1초 정지
    r, _ = feed(tr, move + still)
    assert r.state == STATE_DONE
    assert len(r.trajectory) == len(move) + len(still)
    assert r.trajectory[0].tolist() == [100, 200]
    assert r.timestamps[-1] == pytest.approx((len(move) + len(still) - 1) * DT)


def test_save_path_and_stop_separately(tmp_path, monkeypatch):
    import demo

    monkeypatch.setattr(demo, "RECORDINGS", tmp_path)
    tr = make_tracker()
    r = busy_result(tr)
    path_file, stop_file = demo.save_result(r)
    assert path_file != stop_file
    data = np.loadtxt(path_file, delimiter=",", skiprows=1)
    assert data.shape == (len(r.trajectory), 3)
    assert np.array_equal(data[:, 1:].astype(int), r.trajectory)
    stop = np.loadtxt(stop_file, delimiter=",", skiprows=1)
    assert tuple(stop.astype(int)) == r.stop_position


def test_stop_position_is_where_hand_stopped_not_whole_path():
    tr = make_tracker()
    move = [(100 + i * 20, 100 + i * 10) for i in range(40)]   # 크게 이동
    still = [(1000 + (i % 3) * 3, 600 + (i % 2) * 4) for i in range(31)]   # 마지막에 (1003, 602) 근처에서 정지
    r, _ = feed(tr, move + still)
    assert r.state == STATE_DONE
    sx, sy = r.stop_position
    assert abs(sx - 1003) <= 1 and abs(sy - 602) <= 1
    # 경로 전체 box와는 다름
    assert r.box == (100, 100, 1006, 604)


def test_stop_position_cleared_on_reset_and_none_before_done():
    tr = make_tracker()
    r, t = feed(tr, [(100 + i * 20, 300) for i in range(10)])
    assert r.state == STATE_RECORDING and r.stop_position is None
    r, t = feed(tr, [(500, 300)] * 31, t0=t)
    assert r.stop_position is not None
    tr.reset()
    assert tr.update(None, HAND_NONE, t + 1).stop_position is None


def test_render_shows_stop_position_not_path_box():
    tr = make_tracker()
    r = busy_result(tr)
    shown = tr.render(r)
    r.box = None
    assert np.array_equal(tr.render(r), shown)       # 경로 box는 그리지 않음
    r.stop_position = None
    assert not np.array_equal(tr.render(r), shown)   # 멈춘 위치 표시는 그림


def test_retry_button_bar():
    import demo

    bar = demo.make_button_bar(W + 320, done=True)
    assert bar.shape == (demo.BAR_H, W + 320, 3)
    x1, y1, x2, y2 = demo.BUTTON
    assert demo.in_button((x1 + x2) // 2, (y1 + y2) // 2)
    assert not demo.in_button(x2 + 5, y1)
    assert not demo.in_button(x1, -10)   # 버튼 줄 위(문제 이미지) 클릭은 무시


def test_render_hint_panel_shows_resized_hint():
    import cv2

    tr = make_tracker()
    canvas = tr.render(tr.update(None, HAND_NONE, 0.0))
    x0, y0, nw, nh = tr.hint_rect
    expected = cv2.resize(tr.hint_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(canvas[y0:y0 + nh, W + x0:W + x0 + nw], expected)
    assert nw / nh == pytest.approx(160 / 90, rel=0.02)   # 비율 유지
    assert abs((320 - nw) - 2 * x0) <= 1                   # 가로 가운데


def test_render_does_not_modify_originals():
    problem, hint = make_images()
    p0, h0 = problem.copy(), hint.copy()
    tr = HandTracker(problem, hint, alpha=1.0)
    tr.render(busy_result(tr))
    assert np.array_equal(problem, p0)
    assert np.array_equal(hint, h0)


def test_custom_panel_width():
    tr = make_tracker(panel_width=400)
    assert tr.render(tr.update(None, HAND_NONE, 0.0)).shape == (H, W + 400, 3)


def test_set_images_replaces_and_resets():
    tr = make_tracker()
    feed(tr, [(300, 300)] * 40)
    problem = np.zeros((H, W, 3), np.uint8)
    hint = np.full((300, 100, 3), 255, np.uint8)
    tr.set_images(problem, hint)
    r = tr.update(None, HAND_NONE, 10.0)
    assert r.state == STATE_IDLE and len(r.trajectory) == 0
    canvas = tr.render(r)
    assert np.array_equal(canvas[100:, :W][canvas[100:, :W] != 0], [])   # 새 문제 이미지(검정)
    x0, y0, nw, nh = tr.hint_rect
    assert nh > nw   # 세로로 긴 힌트도 비율 유지
