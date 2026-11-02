import numpy as np
import pytest

from cv.problem_gen.blending import laplacian_pyramid, max_levels, pyramid_blend, reconstruct

ODD_SIZES = [(479, 641), (257, 333), (37, 1), (1, 1)]


@pytest.mark.parametrize("h, w", ODD_SIZES)
def test_reconstruct_returns_original(h, w):
    img = np.random.default_rng(0).integers(0, 256, (h, w, 3)).astype(np.float32)
    lap = laplacian_pyramid(img, max_levels(h, w))
    assert np.allclose(reconstruct(lap), img, atol=1e-3)


@pytest.mark.parametrize("h, w", ODD_SIZES)
def test_blend_keeps_size_and_dtype(h, w):
    rng = np.random.default_rng(0)
    fg = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    bg = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    out = pyramid_blend(fg, bg, rng.random((h, w), dtype=np.float32), levels=6)
    assert out.shape == (h, w, 3) and out.dtype == np.uint8


def test_blend_mask_extremes():
    rng = np.random.default_rng(0)
    fg = rng.integers(0, 256, (479, 641, 3), dtype=np.uint8)
    bg = rng.integers(0, 256, (479, 641, 3), dtype=np.uint8)
    ones = np.ones((479, 641), np.float32)
    assert np.abs(pyramid_blend(fg, bg, ones, 4).astype(int) - fg).max() <= 1
    assert np.abs(pyramid_blend(fg, bg, ones * 0, 4).astype(int) - bg).max() <= 1
