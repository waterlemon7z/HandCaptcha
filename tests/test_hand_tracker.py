"""웹캠 없이 가짜 좌표 시퀀스로 스무딩/dwell/궤적 로직을 검증한다."""
import pytest

from hand_tracker import (
    STATUS_DWELLING,
    STATUS_NO_HAND,
    STATUS_SELECTED,
    STATUS_TRACKING,
    DwellSelector,
    EMASmoother,
    HandTracker,
)

W, H = 400, 300
FPS = 30
DT = 1.0 / FPS


def make_tracker(**kw):
    # 기본: 스무딩 없음, 활성 영역 = 카메라 전체 → 정규화 좌표와 캔버스 좌표가 단순 비례
    kw.setdefault("smoothing", 1.0)
    kw.setdefault("active_margin", 0.0)
    return HandTracker((W, H), **kw)


def norm(tracker, x, y):
    """캔버스 좌표 → 정규화 좌표 (active_margin=0 가정)."""
    cw, ch = tracker.canvas_size
    return (x / (cw - 1), y / (ch - 1))


def feed(tracker, points, t0=0.0):
    """캔버스 좌표 목록을 30fps로 넣고 마지막 상태와 다음 시각을 돌려준다."""
    state, t = None, t0
    for p in points:
        state = tracker.update_point(None if p is None else norm(tracker, *p), t)
        t += DT
    return state, t


# ---------------------------------------------------------------- EMA

def test_ema_first_value_passes_through():
    s = EMASmoother(0.4)
    assert s.update((10.0, 20.0)) == (10.0, 20.0)


def test_ema_formula():
    s = EMASmoother(0.4)
    s.update((0.0, 0.0))
    x, y = s.update((10.0, 100.0))
    assert x == pytest.approx(4.0)
    assert y == pytest.approx(40.0)


def test_ema_converges_to_constant_input():
    s = EMASmoother(0.4)
    s.update((0.0, 0.0))
    for _ in range(50):
        p = s.update((100.0, 50.0))
    assert p == pytest.approx((100.0, 50.0), abs=1e-6)


def test_ema_rejects_bad_alpha():
    with pytest.raises(ValueError):
        EMASmoother(0.0)


def test_smoothing_reduces_jitter():
    tr = make_tracker(smoothing=0.3)
    jitter = [(200 + (8 if i % 2 else -8), 150) for i in range(30)]
    feed(tr, jitter)
    xs = [x for x, _, _ in tr.get_trajectory()[10:]]
    assert max(xs) - min(xs) < 16 * 0.5  # 원래 진폭 16px의 절반 미만


# ---------------------------------------------------------------- 좌표 변환

def test_canvas_size_includes_gap():
    tr = HandTracker((W, H), gap=20)
    assert tr.canvas_size == (2 * W + 20, H)


def test_active_margin_maps_center_region_to_full_canvas():
    tr = HandTracker((W, H), active_margin=0.1)
    cw, ch = tr.canvas_size
    assert tr.normalized_to_canvas((0.1, 0.1)) == pytest.approx((0, 0))
    assert tr.normalized_to_canvas((0.9, 0.9)) == pytest.approx((cw - 1, ch - 1))
    # 활성 영역 밖은 가장자리로 고정
    assert tr.normalized_to_canvas((0.0, 1.0)) == pytest.approx((0, ch - 1))


def test_canvas_to_image_left_and_right():
    tr = HandTracker((W, H), gap=20)
    assert tr.canvas_to_image((50, 60)) == ("left", 50, 60)
    assert tr.canvas_to_image((W + 20 + 50, 60)) == ("right", 50, 60)


def test_canvas_to_image_gap_snaps_to_nearest_edge():
    tr = HandTracker((W, H), gap=20)
    assert tr.canvas_to_image((W + 5, 10)) == ("left", W - 1, 10)
    assert tr.canvas_to_image((W + 15, 10)) == ("right", 0, 10)


# ---------------------------------------------------------------- dwell

def test_dwell_selects_after_dwell_time():
    tr = make_tracker()
    state, _ = feed(tr, [(100, 100)] * FPS)        # 0 ~ 0.967초
    assert not state.selected
    assert state.status == STATUS_DWELLING
    state, _ = feed(tr, [(100, 100)] * 2, t0=FPS * DT)  # 1.0초 도달
    assert state.selected
    assert state.progress == 1.0
    assert state.status == STATUS_SELECTED
    assert state.answer == (100, 100)
    assert state.side == "left"


def test_dwell_progress_is_proportional():
    tr = make_tracker()
    state, _ = feed(tr, [(100, 100)] * 16)  # 마지막 t = 0.5초
    assert state.progress == pytest.approx(0.5)


def test_first_frame_is_tracking_not_dwelling():
    tr = make_tracker()
    state = tr.update_point(norm(tr, 100, 100), 0.0)
    assert state.progress == 0.0
    assert state.status == STATUS_TRACKING


def test_moving_out_of_radius_restarts_dwell():
    tr = make_tracker()
    state, t = feed(tr, [(100, 100)] * 20)
    assert state.progress > 0.5
    state, _ = feed(tr, [(200, 100)], t0=t)  # 100px 이동
    assert state.progress == 0.0


def test_jitter_inside_radius_still_selects():
    tr = make_tracker()
    pts = [(300 + (10 if i % 2 else -10), 120 + (i % 3) * 5) for i in range(40)]
    state, _ = feed(tr, pts)
    assert state.selected
    ax, ay = state.answer
    assert abs(ax - 300) <= 2 and abs(ay - 125) <= 3  # 머문 점들의 평균


def test_custom_dwell_settings():
    tr = make_tracker(dwell_time=0.5, dwell_radius=5)
    state, _ = feed(tr, [(100, 100)] * 16)
    assert state.selected
    tr.reset()
    state, _ = feed(tr, [(100 + (i % 2) * 10, 100) for i in range(40)])  # 반경 5 밖으로 왔다 갔다
    assert not state.selected


def test_answer_on_right_image_uses_original_coords():
    tr = make_tracker(gap=20)
    state, _ = feed(tr, [(W + 20 + 70, 80)] * 40)
    assert state.selected
    assert state.side == "right"
    assert state.answer == (70, 80)


def test_dwell_selector_rejects_bad_params():
    with pytest.raises(ValueError):
        DwellSelector(0, 1.0)


# ---------------------------------------------------------------- 손 사라짐

def test_hand_lost_resets_dwell_after_grace():
    tr = make_tracker(lost_grace=0.15)
    state, t = feed(tr, [(100, 100)] * 20)
    assert state.progress > 0.5
    state, t = feed(tr, [None] * 10, t0=t)  # 0.33초 동안 손 없음
    assert state.point is None
    assert not state.hand_visible
    assert state.progress == 0.0
    assert state.status == STATUS_NO_HAND
    state, _ = feed(tr, [(100, 100)], t0=t)
    assert state.progress == 0.0  # 처음부터 다시


def test_short_loss_within_grace_keeps_state():
    tr = make_tracker(lost_grace=0.15)
    state, t = feed(tr, [(100, 100)] * 20)
    before = state.progress
    state, t = feed(tr, [None] * 2, t0=t)  # 0.067초
    assert state.hand_visible
    assert state.progress == before
    state, _ = feed(tr, [(100, 100)] * 12, t0=t)
    assert state.selected


def test_no_hand_from_start():
    tr = make_tracker()
    state, _ = feed(tr, [None] * 5)
    assert state.point is None
    assert state.status == STATUS_NO_HAND
    assert tr.get_trajectory() == []


# ---------------------------------------------------------------- 궤적

def test_trajectory_records_visible_frames_with_relative_time():
    tr = make_tracker()
    pts = [(10 * i, 100) for i in range(5)] + [None] * 10 + [(60, 100)]
    feed(tr, pts, t0=123.0)  # 시작 시각과 무관하게 0부터
    traj = tr.get_trajectory()
    assert len(traj) == 6
    assert traj[0][2] == pytest.approx(0.0)
    assert traj[-1][2] == pytest.approx(15 * DT)
    ts = [p[2] for p in traj]
    assert ts == sorted(ts)
    assert traj[2][:2] == pytest.approx((20, 100))


def test_trajectory_stops_after_selection():
    tr = make_tracker()
    state, t = feed(tr, [(100, 100)] * 40)
    assert state.selected
    n = len(tr.get_trajectory())
    state, _ = feed(tr, [(300, 200)] * 10, t0=t)
    assert len(tr.get_trajectory()) == n
    assert state.selected and state.answer == (100, 100)


def test_get_trajectory_returns_copy():
    tr = make_tracker()
    feed(tr, [(100, 100)] * 3)
    tr.get_trajectory().clear()
    assert len(tr.get_trajectory()) == 3


def test_reset_clears_everything():
    tr = make_tracker()
    feed(tr, [(100, 100)] * 40)
    tr.reset()
    assert tr.get_trajectory() == []
    assert tr.state.status == STATUS_NO_HAND
    state, _ = feed(tr, [(100, 100)], t0=50.0)
    assert not state.selected
    assert tr.get_trajectory()[0][2] == 0.0


def test_uses_clock_when_t_omitted():
    now = [10.0]
    tr = make_tracker(clock=lambda: now[0])
    for _ in range(31):
        state = tr.update_point(norm(tr, 100, 100))
        now[0] += DT
    assert state.selected


def test_unknown_override_raises():
    with pytest.raises(TypeError):
        HandTracker((W, H), dwel_time=1.0)
