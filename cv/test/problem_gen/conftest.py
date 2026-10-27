import cv2
import numpy as np
import pytest

# 테스트 물체는 배경(회색조)에 없는 순수 빨간색이라 색만으로 위치를 찾을 수 있다
RED = (0, 0, 255)


@pytest.fixture
def asset_dir(tmp_path):
    """회색 바닥에 어두운 물건(사각형) 4개가 놓인 배경 2장과 빨간 물체 3개(원, 사각형, 삼각형)로 된 가짜 에셋 폴더.

    물건 사각형은 물체(긴 변 최대 약 115px)보다 크고 서로 140px 이상 떨어져 있어,
    물체가 물건 밑에 깔려도 보이는 부분이 두 조각으로 나뉘지 않는다.
    """
    rng = np.random.default_rng(0)
    (tmp_path / "backgrounds").mkdir()
    (tmp_path / "objects").mkdir()
    items = [(0.10, 0.10, 0.30, 0.40), (0.45, 0.15, 0.62, 0.45), (0.75, 0.55, 0.95, 0.90), (0.15, 0.60, 0.38, 0.88)]
    for i, (w, h) in enumerate([(1600, 900), (1280, 720)]):
        gray = np.clip(200 + rng.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)
        for x1, y1, x2, y2 in items:
            gray[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 60
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


def always_fail(image, target):
    """항상 못 찾는 가짜 validator."""
    return []


def always_solve(image, target):
    """항상 맞히는 가짜 validator (테스트 전용). 빨간 덩어리(정답과 방해용 물체)마다 박스 (x1, y1, x2, y2)를 반환한다."""
    b, g, r = (image[..., i].astype(int) for i in range(3))
    red = (r - np.maximum(g, b) > 100).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(red)
    return [(x, y, x + w, y + h) for x, y, w, h, area in stats[1:n] if area > 50]
