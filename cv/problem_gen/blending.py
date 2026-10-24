"""라플라시안 피라미드 블렌딩 (pyrDown / pyrUp 직접 구현).

pyrDown은 크기를 ((w+1)//2, (h+1)//2)로 줄이므로, pyrUp할 때 항상 dstsize로
위 단계 크기를 지정해 홀수 크기에서도 단계 간 크기가 어긋나지 않게 한다.
라플라시안은 음수가 생기므로 float32로 계산한다.
"""
import cv2
import numpy as np


def _size(img):
    return img.shape[1], img.shape[0]


def max_levels(h, w):
    """한 변이 1픽셀이 될 때까지 줄일 수 있는 단계 수."""
    return int(np.log2(min(h, w)))


def gaussian_pyramid(img, levels):
    pyr = [img.astype(np.float32)]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_pyramid(img, levels):
    gauss = gaussian_pyramid(img, levels)
    lap = [fine - cv2.pyrUp(coarse, dstsize=_size(fine)) for fine, coarse in zip(gauss[:-1], gauss[1:])]
    lap.append(gauss[-1])
    return lap


def reconstruct(lap):
    img = lap[-1]
    for fine in reversed(lap[:-1]):
        img = cv2.pyrUp(img, dstsize=_size(fine)) + fine
    return img


def pyramid_blend(fg, bg, mask, levels):
    """mask(H×W, 0~1)가 1인 곳은 fg, 0인 곳은 bg를 쓰도록 단계별로 섞는다."""
    levels = min(levels, max_levels(*mask.shape[:2]))
    lap_fg = laplacian_pyramid(fg, levels)
    lap_bg = laplacian_pyramid(bg, levels)
    gauss_mask = gaussian_pyramid(mask, levels)
    blended = []
    for lf, lb, gm in zip(lap_fg, lap_bg, gauss_mask):
        if lf.ndim == 3:
            gm = gm[..., None]
        blended.append(lf * gm + lb * (1 - gm))
    return np.clip(np.rint(reconstruct(blended)), 0, 255).astype(np.uint8)
