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
    crop_around,
    index_tip,
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


def test_index_tip_is_landmark_8():
    lm = make_landmarks(0.3, 0.6)
    assert index_tip(lm) == pytest.approx((0.24, 0.44))   # 펴진 검지 끝 = (cx-0.06, cy-0.16)


def test_cursor_follows_index_tip_with_mirror():
    tr = make_tracker()
    r = tr.process_landmarks(make_landmarks(0.25, 0.5), t=0.0)
    # 검지 끝 카메라 x=0.19 → 좌우 반전 → 화면 x=0.81, y=0.34
    assert r.position == (round(0.81 * (W - 1)), round(0.34 * (H - 1)))
    assert r.hand_status == HAND_OPEN


def test_landmarks_without_mirror():
    tr = make_tracker(mirror=False)
    r = tr.process_landmarks(make_landmarks(0.25, 0.5), t=0.0)
    assert r.position[0] == round(0.19 * (W - 1))


def test_palm_anchor_option():
    tr = make_tracker(anchor="palm")
    r = tr.process_landmarks(make_landmarks(0.25, 0.5), t=0.0)
    assert r.position == (round(0.75 * (W - 1)), round(0.5 * (H - 1)))


def test_bad_anchor_and_clip_mode():
    with pytest.raises(ValueError):
        make_tracker(anchor="thumb")
    with pytest.raises(ValueError):
        make_tracker(clip_mode="full")


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


# ---------------------------------------------------------------- EMA 전 원본 좌표 (raw)

def test_raw_position_and_trajectory_are_before_ema():
    tr = make_tracker(alpha=0.5)
    pts = [(100, 100), (200, 300), (300, 100), (400, 300)]
    r, _ = feed(tr, pts)
    assert r.state == STATE_RECORDING
    assert r.raw_trajectory.tolist() == [list(p) for p in pts]          # 원본 그대로
    assert r.trajectory.tolist() != r.raw_trajectory.tolist()           # 보정본은 다름
    assert r.trajectory.shape == r.raw_trajectory.shape                 # 같은 프레임끼리 짝
    assert r.raw_position == (400, 300)
    assert r.position != r.raw_position


def test_raw_keeps_float_without_rounding():
    tr = make_tracker(alpha=0.5)
    pts = [(100.4, 200.6), (150.25, 210.75), (199.5, 220.5)]
    r, _ = feed(tr, pts)
    assert r.raw_trajectory.dtype == np.float64
    assert r.raw_trajectory.tolist() == [list(p) for p in pts]          # 소수점 그대로
    assert r.raw_position == (199.5, 220.5)
    assert all(isinstance(v, float) for v in r.raw_position)
    # 보정본(position, trajectory)은 팀 약속대로 정수
    assert r.trajectory.dtype == np.int32
    assert all(isinstance(v, int) for v in r.position)


def test_raw_trajectory_float_dtype_even_when_empty():
    tr = make_tracker()
    r = tr.update(None, HAND_NONE, 0.0)
    assert r.raw_trajectory.shape == (0, 2) and r.raw_trajectory.dtype == np.float64


def test_raw_from_landmarks_is_not_rounded():
    """랜드마크 → 문제 이미지 변환 결과가 정수가 아니어도 원본은 그대로 남는다."""
    tr = make_tracker()
    lm = make_landmarks(0.3, 0.5, folded=4)
    r = tr.process_landmarks(lm, 0.0)
    tx, ty = index_tip(lm)
    expected = ((1 - tx) * (W - 1), ty * (H - 1))
    assert r.raw_position == pytest.approx(expected)
    assert r.raw_trajectory[0] == pytest.approx(expected)
    assert expected[0] != round(expected[0])                             # 실제로 소수 좌표인 경우


def test_render_accepts_float_coordinates():
    """그릴 때만 int로 바꾸므로 float 좌표가 들어와도 render가 동작한다."""
    tr = make_tracker()
    r = busy_result(tr)
    r.position = (300.6, 200.4)
    r.stop_position = (500.5, 400.49)
    as_float = tr.render(r)
    r.position, r.stop_position = (301, 200), (500, 400)
    assert np.array_equal(as_float, tr.render(r))


def test_raw_is_none_when_hand_ignored():
    tr = make_tracker()
    r = tr.update((W + 50, 300), HAND_OPEN, 0.0)   # 힌트 패널 쪽
    assert r.position is None and r.raw_position is None


def test_raw_cleared_on_cancel_and_reset():
    tr = make_tracker()
    r, t = feed(tr, [(100 + i * 10, 300) for i in range(5)])
    r = tr.update(None, HAND_NONE, t + 1.0)           # 손 놓침 → 취소
    assert len(r.raw_trajectory) == 0
    feed(tr, [(100, 300)] * 3, t0=t + 2)
    tr.reset()
    assert len(tr.update(None, HAND_NONE, t + 5).raw_trajectory) == 0


def test_raw_clipped_into_image():
    """EMA 보정본은 이미지 안이지만 원본은 살짝 밖인 프레임도 이미지 범위로 잘라 기록한다."""
    tr = make_tracker(alpha=0.5)
    tr.update((W - 10, 300), HAND_FIST, 0.0)
    r = tr.update((W + 4, 300), HAND_FIST, DT)       # 보정본 x = W-3 (안), 원본 x = W+4 (밖)
    assert r.position[0] == W - 3
    assert r.raw_position[0] == W - 1.0
    assert r.raw_trajectory[-1, 0] == W - 1.0


# ---------------------------------------------------------------- 카메라 프레임 크기

def test_frame_size_from_process_landmarks():
    tr = make_tracker()
    r = tr.process_landmarks(make_landmarks(), 0.0, np.zeros((480, 640, 3), np.uint8))
    assert r.frame_size == (640, 480)                                   # (w, h) 순서


def test_frame_size_none_without_frame():
    tr = make_tracker()
    assert tr.process_landmarks(make_landmarks(), 0.0).frame_size is None
    assert tr.update((100, 100), HAND_OPEN, DT).frame_size is None


def test_frame_size_reported_even_without_hand_and_after_reset():
    tr = make_tracker()
    r = tr.process_landmarks(None, 0.0, np.zeros((720, 1280, 3), np.uint8))
    assert r.position is None and r.frame_size == (1280, 720)
    tr.reset()
    assert tr.update(None, HAND_NONE, 1.0).frame_size == (1280, 720)   # 카메라 정보라 reset해도 유지


def test_frame_size_follows_latest_frame():
    tr = make_tracker()
    tr.process_landmarks(None, 0.0, np.zeros((480, 640, 3), np.uint8))
    r = tr.process_landmarks(None, DT, np.zeros((240, 320, 3), np.uint8))
    assert r.frame_size == (320, 240)


# ---------------------------------------------------------------- 기록 중 프레임 버퍼

CAM_H, CAM_W = 480, 640


def make_frame():
    """좌표마다 값이 다른 가짜 카메라 프레임 (crop 위치를 확인하려고)."""
    ys, xs = np.mgrid[0:CAM_H, 0:CAM_W]
    return np.stack([xs % 256, ys % 256, (xs // 256) * 64 + (ys // 256)], axis=2).astype(np.uint8)


def run_with_frames(tr, n_move=20, n_still=31):
    """검지 끝을 움직였다가 멈추는 손을 frame과 함께 넣는다."""
    frame, t, r = make_frame(), 0.0, None
    for i in range(n_move + n_still):
        cx = 0.3 + 0.01 * min(i, n_move)
        r = tr.process_landmarks(make_landmarks(cx, 0.5, folded=0 if i else 4), t, frame)
        t += DT
    return r, frame


def test_frames_returned_only_when_done():
    tr = make_tracker()
    frame = make_frame()
    r = tr.process_landmarks(make_landmarks(0.4, 0.5, folded=4), 0.0, frame)
    assert r.state == STATE_RECORDING
    assert r.frames is None and r.frame_timestamps is None
    r, _ = run_with_frames(make_tracker())
    assert r.state == STATE_DONE
    assert r.frames is not None


def test_frames_align_with_trajectory():
    r, _ = run_with_frames(make_tracker())
    assert len(r.frames) == len(r.trajectory) == len(r.frame_timestamps)
    assert np.allclose(r.frame_timestamps, r.timestamps)
    assert all(f.shape == (128, 128, 3) for f in r.frames)


def test_crop_is_centered_on_index_tip_and_mirrored():
    tr = make_tracker(crop_size=64)
    frame = make_frame()
    lm = make_landmarks(0.5, 0.5, folded=4)       # 접힌 검지 끝 = (0.44, 0.52)
    r = tr.process_landmarks(lm, 0.0, frame)
    for _ in range(31):
        r = tr.process_landmarks(lm, r.timestamps[-1] + DT, frame)
    assert r.state == STATE_DONE
    tx, ty = round(0.44 * (CAM_W - 1)), round(0.52 * (CAM_H - 1))
    expected = frame[ty - 32:ty + 32, tx - 32:tx + 32][:, ::-1]   # 화면과 같은 방향(좌우 반전)
    assert np.array_equal(r.frames[0], expected)


def test_crop_keeps_size_at_frame_edge():
    frame = make_frame()
    crop = crop_around(frame, (2, CAM_H - 3), 128)
    assert crop.shape == (128, 128, 3)
    assert crop[0, 0].tolist() == [0, 0, 0]                  # 프레임 밖은 검정
    assert crop[64, 64].tolist() == frame[CAM_H - 3, 2].tolist()   # 가운데 = 지정한 점


def test_quarter_mode():
    r, frame = run_with_frames(make_tracker(clip_mode="quarter"))
    assert r.frames[0].shape == (CAM_H // 4, CAM_W // 4, 3)


def test_clip_mode_none_collects_nothing():
    r, _ = run_with_frames(make_tracker(clip_mode=None))
    assert r.state == STATE_DONE
    assert r.frames == [] and len(r.frame_timestamps) == 0


def test_frames_cleared_on_cancel():
    tr = make_tracker()
    frame = make_frame()
    tr.process_landmarks(make_landmarks(0.4, 0.5, folded=4), 0.0, frame)
    tr.process_landmarks(make_landmarks(0.41, 0.5), DT, frame)
    tr.process_landmarks(None, 1.0)                            # 0.5초 넘게 놓침 → 취소
    assert tr._clips == [] and tr._clip_times == []


def test_no_frames_collected_in_idle_or_after_done():
    tr = make_tracker()
    frame = make_frame()
    for i in range(10):                                       # 편 손: idle
        tr.process_landmarks(make_landmarks(0.4, 0.5), i * DT, frame)
    assert tr._clips == []
    r, _ = run_with_frames(tr)
    n = len(r.frames)
    r = tr.process_landmarks(make_landmarks(0.6, 0.5), 10.0, frame)   # done 이후
    assert len(r.frames) == n
