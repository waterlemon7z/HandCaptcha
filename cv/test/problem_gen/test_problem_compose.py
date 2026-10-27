import cv2
import numpy as np

from cv.problem_gen.compose import SURROUND_PX, fit_background, floor_map, place_object, transform_object


def _circle(size=100):
    obj = np.zeros((size, size, 4), np.uint8)
    cv2.circle(obj, (size // 2, size // 2), size // 2 - 5, (0, 0, 255, 255), -1)
    return obj


def test_fit_background_crops_to_output_size():
    for h, w in [(720, 1280), (900, 900), (1067, 1600), (721, 1281)]:
        assert fit_background(np.zeros((h, w, 3), np.uint8), (1280, 720)).shape == (720, 1280, 3)


def test_transform_object_scales_long_side():
    obj = np.zeros((100, 400, 4), np.uint8)
    obj[..., 3] = 255
    out = transform_object(obj, np.random.default_rng(0), 80, (0, 0))
    assert out.shape == (20 + 2, 80 + 2, 4)  # 투명 테두리 1px 포함
    assert cv2.boundingRect(out[..., 3]) == (1, 1, 80, 20)


def test_transform_object_rotation_keeps_whole_object():
    # 회전해도 캔버스를 넓히므로 물체가 잘리지 않는다: 면적이 유지되고 가장자리는 투명
    obj = np.zeros((60, 200, 4), np.uint8)
    obj[..., :3] = (0, 200, 0)
    obj[..., 3] = 255
    for seed in range(5):
        out = transform_object(obj, np.random.default_rng(seed), 200, (30, 60))
        assert abs(out[..., 3].sum() / 255 - 60 * 200) / (60 * 200) < 0.03
        border = np.concatenate([out[0, :, 3], out[-1, :, 3], out[:, 0, 3], out[:, -1, 3]])
        assert border.max() < 128
        # 투명 픽셀 색이 섞이지 않아 반투명 테두리도 원래 색을 유지한다
        edge = (out[..., 3] > 20) & (out[..., 3] < 235)
        assert np.abs(out[edge][:, :3].astype(int) - (0, 200, 0)).max() <= 3


def test_place_object_hides_next_to_clutter():
    # 왼쪽 60%는 빈 바닥, 오른쪽 40%는 물건 테두리가 빽빽한 배경
    # → 물체는 빽빽한 쪽 위에 올라가지 않고, 주변 고리가 빽빽한 쪽에 닿는 경계 근처에 놓인다
    bg = np.full((480, 640, 3), 200, np.uint8)
    for x in range(384, 640, 12):
        cv2.line(bg, (x, 0), (x, 479), (30, 30, 30), 2)
    for seed in range(20):
        _, alpha = place_object(bg, _circle(), np.random.default_rng(seed), 0.05, 100, 3)
        assert not alpha[:, 384:].any(), seed
        x, _, w, _ = cv2.boundingRect((alpha > 0.5).astype(np.uint8))
        assert x + w > 384 - SURROUND_PX, seed


def test_place_object_avoids_occupied():
    bg = np.full((480, 640, 3), 200, np.uint8)
    occupied = np.zeros((480, 640), bool)
    occupied[:, :320] = True
    for seed in range(20):
        _, alpha = place_object(bg, _circle(), np.random.default_rng(seed), 0.05, 30, 5, occupied)
        assert not alpha[:, :320].any(), seed


def test_place_object_same_seed_same_position():
    bg = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8), (7, 7), 0)
    a1 = place_object(bg, _circle(), np.random.default_rng(4), 0.05, 30, 5)[1]
    a2 = place_object(bg, _circle(), np.random.default_rng(4), 0.05, 30, 5)[1]
    assert np.array_equal(a1, a2)


def test_floor_map_marks_items():
    bg = np.full((480, 640, 3), 200, np.uint8)
    bg[100:300, 400:600] = (40, 90, 160)
    floor = floor_map(bg)
    assert floor[50, 50] > 0.99 and floor[200, 500] < 0.01


def test_place_object_tucks_under_items():
    # 왼쪽 60%는 바닥, 오른쪽 40%는 사진 속 물건: 물체가 경계에 걸쳐 물건 쪽 부분이 가려진다
    # 경계가 한 줄뿐이라 조건에 맞는 후보가 드물어 후보를 300개 뽑는다 (실제 사진은 100개로 충분)
    bg = np.full((480, 640, 3), 200, np.uint8)
    bg[:, 384:] = (40, 90, 160)
    floor = floor_map(bg)
    full = np.count_nonzero(_circle()[..., 3] > 127)
    for seed in range(10):
        _, alpha = place_object(bg, _circle(), np.random.default_rng(seed), 0.05, 300, 3, None, floor, (0.2, 0.45))
        visible = alpha > 0.5
        assert not visible[:, 386:].any(), seed
        assert 0.15 <= 1 - np.count_nonzero(visible) / full <= 0.5, seed


def test_place_object_without_tuck_spot_keeps_whole_object():
    # 물건이 없는 배경이면 가릴 곳이 없으므로 물체를 그대로 놓는다
    bg = np.full((480, 640, 3), 200, np.uint8)
    _, alpha = place_object(bg, _circle(), np.random.default_rng(0), 0.05, 100, 3, None, floor_map(bg), (0.2, 0.45))
    assert np.count_nonzero(alpha > 0.5) == np.count_nonzero(_circle()[..., 3] > 127)
