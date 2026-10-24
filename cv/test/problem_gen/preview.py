"""생성된 문제를 눈으로 확인하는 스크립트 (pytest 대상 아님).

    python -m cv.test.problem_gen.preview --out <폴더> --seeds 0 1 2

seed마다 [왼쪽 | 오른쪽 | 오른쪽 + 정답 영역] 이미지와,
블렌딩 전후 비교(물체 주변 확대) 이미지를 저장한다.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from cv.problem_gen import ProblemGenerator
from cv.problem_gen import generator as G
from cv.problem_gen.assets import list_assets, read_image, read_object
from cv.problem_gen.blending import pyramid_blend
from cv.problem_gen.compose import fit_background, place_object


def save_problem(gen, seed, out):
    problem_id, left, right = gen.create(seed=seed)
    # 정답 영역은 판정 후에만 공개되므로 아무 점으로 판정해서 받는다
    result = gen.solve(problem_id, (0, 0))
    overlay = right.copy()
    contours, _ = cv2.findContours(result["answer_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    x, y, w, h = result["answer_box"]
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(str(out / f"problem_{seed:03d}.png"), np.hstack([left, right, overlay]))


def save_blend_comparison(asset_dir, seed, out):
    """같은 배치에서 alpha로만 붙인 결과와 피라미드 블렌딩 결과를 확대해 나란히 저장한다."""
    rng = np.random.default_rng(seed)
    backgrounds, objects = list_assets(asset_dir)
    bg = fit_background(read_image(backgrounds[rng.integers(len(backgrounds))]), G.OUTPUT_SIZE)
    obj = read_object(objects[rng.integers(len(objects))])
    composite, alpha = place_object(bg, obj, rng, G.OBJECT_SCALE_RANGE, G.PLACE_MARGIN)
    blended = pyramid_blend(composite, bg, alpha, G.BLEND_LEVELS)

    x, y, w, h = cv2.boundingRect(np.where(alpha > 0.5, 255, 0).astype(np.uint8))
    pad = 20
    crop = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
    zoom = [cv2.resize(img[crop], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST) for img in (composite, blended)]
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
