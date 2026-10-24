import cv2
import numpy as np
import pytest

# 테스트 물체는 배경(회색조)에 없는 순수 빨간색이라 색만으로 위치를 찾을 수 있다
RED = (0, 0, 255)


@pytest.fixture
def asset_dir(tmp_path):
    """회색조 질감 배경 2장과 빨간 원 물체 1개로 된 가짜 에셋 폴더."""
    rng = np.random.default_rng(0)
    (tmp_path / "backgrounds").mkdir()
    (tmp_path / "objects").mkdir()
    for i, (w, h) in enumerate([(800, 600), (1280, 720)]):
        gray = cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8), (7, 7), 0)
        cv2.imwrite(str(tmp_path / "backgrounds" / f"bg_{i}.png"), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    obj = np.zeros((200, 200, 4), np.uint8)
    cv2.circle(obj, (100, 100), 90, (*RED, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(tmp_path / "objects" / "red_circle.png"), obj)
    return tmp_path


def always_fail(left_img, right_img):
    """항상 못 찾는 가짜 validator."""
    return None


def always_solve(left_img, right_img):
    """항상 맞히는 가짜 validator (테스트 전용). 빨간 픽셀들의 중심을 반환한다."""
    b, g, r = (right_img[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero(r - np.maximum(g, b) > 100)
    return xs.mean(), ys.mean()
