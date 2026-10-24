"""물체 배치와 오른쪽 이미지 변환."""
import cv2
import numpy as np


def fit_background(img, size):
    """가운데를 잘라 size(w, h) 비율에 맞춘 뒤 크기를 맞춘다."""
    w, h = size
    ih, iw = img.shape[:2]
    if iw * h > ih * w:
        cw = ih * w // h
        x0 = (iw - cw) // 2
        img = img[:, x0:x0 + cw]
    else:
        ch = iw * h // w
        y0 = (ih - ch) // 2
        img = img[y0:y0 + ch]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def place_object(bg, obj_bgra, rng, scale_range, margin):
    """물체를 랜덤 크기·위치로 배경에 붙인다.

    물체의 긴 변 = 배경 짧은 변 × scale_range 안의 값.
    margin은 가장자리에서 띄울 비율로, 오른쪽 이미지를 확대·회전해도 물체가 잘리지 않게 한다.
    반환: (alpha로 합성한 이미지 uint8, 배경 크기의 alpha float32 0~1)
    """
    h, w = bg.shape[:2]
    oh, ow = obj_bgra.shape[:2]
    f = rng.uniform(*scale_range) * min(h, w) / max(oh, ow)
    ow, oh = max(1, round(ow * f)), max(1, round(oh * f))
    obj = cv2.resize(obj_bgra, (ow, oh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)

    mx, my = int(w * margin), int(h * margin)
    x0 = int(rng.integers(mx, w - mx - ow + 1))
    y0 = int(rng.integers(my, h - my - oh + 1))

    alpha = np.zeros((h, w), np.float32)
    alpha[y0:y0 + oh, x0:x0 + ow] = obj[..., 3] / 255.0
    fg = bg.astype(np.float32)
    # 투명 픽셀의 색은 의미가 없으므로 alpha로 먼저 섞어 테두리 색 번짐을 막는다
    a = alpha[y0:y0 + oh, x0:x0 + ow, None]
    fg[y0:y0 + oh, x0:x0 + ow] = obj[..., :3] * a + fg[y0:y0 + oh, x0:x0 + ow] * (1 - a)
    return np.clip(np.rint(fg), 0, 255).astype(np.uint8), alpha


def cover_scale(angle_deg, w, h):
    """회전해도 프레임에 빈 모서리가 생기지 않는 최소 확대 배율."""
    t = np.deg2rad(abs(angle_deg))
    return np.cos(t) + max(w / h, h / w) * np.sin(t)


def random_affine(w, h, rng, angle_range, extra_scale_range):
    """이미지 중심 기준 회전·확대 행렬(2×3). 각도 부호는 랜덤."""
    angle = rng.uniform(*angle_range) * rng.choice([-1, 1])
    scale = cover_scale(angle, w, h) * rng.uniform(*extra_scale_range)
    return cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)


def warp(img, mask, M):
    """이미지와 마스크에 같은 변환을 적용한다. 마스크는 0/255가 유지되게 최근접 보간."""
    h, w = img.shape[:2]
    # 확대로 빈 모서리는 없지만 반올림으로 생기는 가장자리 1픽셀을 위해 REFLECT
    img_t = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask_t = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_t, mask_t
