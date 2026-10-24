from pathlib import Path

import cv2
import numpy as np

BACKGROUND_EXTS = {".png", ".jpg", ".jpeg"}


def list_assets(root):
    """(배경 경로 목록, 물체 경로 목록). seed 재현을 위해 이름순으로 정렬한다."""
    root = Path(root)
    backgrounds = sorted(p for p in (root / "backgrounds").iterdir() if p.suffix.lower() in BACKGROUND_EXTS)
    objects = sorted(p for p in (root / "objects").iterdir() if p.suffix.lower() == ".png")
    return backgrounds, objects


def read_image(path, flags=cv2.IMREAD_COLOR):
    # cv2.imread는 Windows에서 한글 경로를 읽지 못하므로 imdecode를 쓴다
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def read_object(path):
    """투명 배경 PNG를 BGRA로 읽는다."""
    img = read_image(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"물체 이미지는 투명 배경(BGRA) PNG여야 합니다: {path}")
    return img
