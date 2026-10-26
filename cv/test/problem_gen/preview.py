"""생성된 문제를 눈으로 확인하는 스크립트 (pytest 대상 아님).

    python -m cv.test.problem_gen.preview --out <폴더> --seeds 0 1 2

seed마다 [합성 이미지 | 합성 이미지 + 정답 영역]과 아래에 힌트(물체 원본) 줄을 붙인 이미지와,
블렌딩 방식별 비교(물체 주변 확대) 이미지를 저장한다.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from cv.problem_gen import ProblemGenerator
from cv.problem_gen import generator as G
from cv.problem_gen.assets import list_assets, read_image, read_object
from cv.problem_gen.compose import fit_background, place_object, transform_object

HINT_HEIGHT = 160


def _hint_row(hints, width):
    """물체 원본을 체크무늬(투명 표시) 위에 같은 높이로 늘어놓는다."""
    tiles = []
    for hint in hints:
        h, w = hint.shape[:2]
        hint = cv2.resize(hint, (max(1, round(w * HINT_HEIGHT / h)), HINT_HEIGHT), interpolation=cv2.INTER_AREA)
        yy, xx = np.indices(hint.shape[:2])
        board = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 230, 190).astype(np.float32)
        a = hint[..., 3:] / 255.0
        tiles.append(np.rint(hint[..., :3] * a + board * (1 - a)).astype(np.uint8))
        tiles.append(np.full((HINT_HEIGHT, 20, 3), 255, np.uint8))
    row = np.hstack(tiles)[:, :width]
    return np.hstack([row, np.full((HINT_HEIGHT, width - row.shape[1], 3), 255, np.uint8)])


def save_problem(gen, seed, out):
    problem_id, image, hints = gen.create(seed=seed)
    # 정답은 문제가 끝나야 공개되므로 화면 밖을 짚어 오답으로 끝낸다
    result = gen.solve(problem_id, (-100, -100))
    overlay = image.copy()
    for mask, (x1, y1, x2, y2) in zip(result["answer_masks"], result["answer_boxes"]):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
    top = np.hstack([image, overlay])
    cv2.imwrite(str(out / f"problem_{seed:03d}.png"), np.vstack([top, _hint_row(hints, top.shape[1])]))


def save_blend_comparison(asset_dir, seed, out):
    """물체 하나를 붙여 [alpha로만 붙임 | 블렌딩(마스크 그대로) | 블렌딩(마스크 흐림)]을 확대해 나란히 저장한다."""
    rng = np.random.default_rng(seed)
    backgrounds, objects = list_assets(asset_dir)
    bg = fit_background(read_image(backgrounds[rng.integers(len(backgrounds))]), G.OUTPUT_SIZE)
    obj = read_object(objects[rng.integers(len(objects))])
    obj = transform_object(obj, rng, rng.uniform(*G.OBJECT_SCALE_RANGE) * min(G.OUTPUT_SIZE), G.OBJECT_ANGLE_RANGE)
    composite, alpha = place_object(bg, obj, rng, G.PLACE_MARGIN, G.PLACE_CANDIDATES, G.PLACE_TOP_K)
    panels = {
        "alpha only": composite,
        "blend": G.blend_object(composite, bg, alpha, feather_sigma=0),
        f"blend + feather {G.BLEND_FEATHER_SIGMA:g}px": G.blend_object(composite, bg, alpha),
    }

    x, y, w, h = cv2.boundingRect(np.where(alpha > 0.5, 255, 0).astype(np.uint8))
    pad = 20
    crop = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
    zoom = []
    for label, img in panels.items():
        img = cv2.resize(img[crop], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        zoom.append(img)
    cv2.imwrite(str(out / f"blend_{seed:03d}.png"), np.hstack(zoom))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--assets", default="assets")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    gen = ProblemGenerator(args.assets)
    for seed in args.seeds:
        save_problem(gen, seed, args.out)
        save_blend_comparison(args.assets, seed, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
