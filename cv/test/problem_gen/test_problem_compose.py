import cv2
import numpy as np

from cv.problem_gen.compose import cover_scale, fit_background, random_affine, warp


def test_fit_background_crops_to_output_size():
    for h, w in [(720, 1280), (900, 900), (481, 641)]:
        assert fit_background(np.zeros((h, w, 3), np.uint8), (640, 480)).shape == (480, 640, 3)


def test_mask_follows_image_transform():
    img = np.zeros((480, 640, 3), np.uint8)
    mask = np.zeros((480, 640), np.uint8)
    cv2.rectangle(img, (300, 200), (380, 260), (0, 255, 0), -1)
    cv2.rectangle(mask, (300, 200), (380, 260), 255, -1)
    M = random_affine(640, 480, np.random.default_rng(3), (3, 6), (1.0, 1.05))
    img_t, mask_t = warp(img, mask, M)

    green = img_t[..., 1] > 128
    inter = np.logical_and(green, mask_t > 0).sum()
    union = np.logical_or(green, mask_t > 0).sum()
    assert inter / union > 0.9
    assert set(np.unique(mask_t)) <= {0, 255}


def test_no_empty_corners_after_rotation():
    # 흰 이미지를 테두리 0으로 변환했을 때 빈 곳이 없으면 확대 배율이 충분한 것
    white = np.full((480, 640), 255, np.uint8)
    for angle in (3, 6, -6):
        M = cv2.getRotationMatrix2D((319.5, 239.5), angle, cover_scale(angle, 640, 480))
        out = cv2.warpAffine(white, M, (640, 480), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        assert (out[1:-1, 1:-1] > 0).all()
