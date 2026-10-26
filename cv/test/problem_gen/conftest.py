import cv2
import numpy as np
import pytest

# 테스트 물체는 배경(회색조)에 없는 순수 빨간색이라 색만으로 위치를 찾을 수 있다
RED = (0, 0, 255)


@pytest.fixture
def asset_dir(tmp_path):
    """회색조 질감 배경 2장과 빨간 물체 3개(원, 사각형, 삼각형)로 된 가짜 에셋 폴더."""
    rng = np.random.default_rng(0)
    (tmp_path / "backgrounds").mkdir()
    (tmp_path / "objects").mkdir()
    for i, (w, h) in enumerate([(1600, 1000), (1280, 720)]):
        gray = cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8), (7, 7), 0)
        cv2.imwrite(str(tmp_path / "backgrounds" / f"bg_{i}.png"), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    shapes = {
        "red_circle": lambda m: cv2.circle(m, (100, 100), 90, (*RED, 255), -1, cv2.LINE_AA),
        "red_square": lambda m: cv2.rectangle(m, (20, 20), (180, 180), (*RED, 255), -1),
        "red_triangle": lambda m: cv2.fillPoly(m, [np.array([[100, 10], [190, 190], [10, 190]])], (*RED, 255),
                                               cv2.LINE_AA),
    }
    for name, draw in shapes.items():
        obj = np.zeros((200, 200, 4), np.uint8)
        draw(obj)
        cv2.imwrite(str(tmp_path / "objects" / f"{name}.png"), obj)
    return tmp_path


def always_fail(image, hints):
    """항상 못 찾는 가짜 validator."""
    return []


def always_solve(image, hints):
    """항상 다 찾는 가짜 validator (테스트 전용). 빨간 덩어리마다 박스 (x1, y1, x2, y2)를 반환한다."""
    b, g, r = (image[..., i].astype(int) for i in range(3))
    red = (r - np.maximum(g, b) > 100).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(red)
    return [(x, y, x + w, y + h) for x, y, w, h, area in stats[1:n] if area > 50]
