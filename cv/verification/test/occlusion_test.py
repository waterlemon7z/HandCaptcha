"""배경·물체·배치를 고정한 인공 가림 비교 실험.

실행:
    python -m cv.verification.test.occlusion_test

기존 생성기 파일과 검수기 파일을 수정하지 않는다.
가림 면적은 변형된 대상의 alpha > 0.5 픽셀 수를 기준으로 한다.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from cv.problem_gen.assets import list_assets, read_image, read_object
from cv.problem_gen.compose import (
    fit_background,
    transform_object,
    place_object,
)
from cv.problem_gen.generator import blend_object

from cv.verification.difficulty_check import DifficultyCheck


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("orb", "sift")
ASSET_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "occlusion_results"

NUM_SAMPLES = 100
START_SEED = 10

# 비교할 가림 비율
OCCLUSION_RATIOS = (0.0, 0.25, 0.40)

# 앞서 확인한 생성기와 같은 크기·회전 범위
OUTPUT_SIZE = (1280, 720)
OBJECT_SCALE_RANGE = (0.10, 0.16)
OBJECT_ANGLE_RANGE = (-45.0, 45.0)

IOU_THRESHOLD = 0.1

# True면 이미지 창 표시. Q 또는 ESC로 종료.
SHOW_WINDOW = False
WAIT_MS = 200


def save_image(path, image):
    """Windows 한글 경로에서도 이미지 저장."""
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError(f"이미지 저장 실패: {path}")

    encoded.tofile(str(path))


def mask_box(mask):
    """마스크를 감싸는 xyxy 박스."""
    x, y, w, h = cv2.boundingRect(
        mask.astype(np.uint8)
    )
    return x, y, x + w, y + h


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    intersection = (
        max(0, min(ax2, bx2) - max(ax1, bx1))
        * max(0, min(ay2, by2) - max(ay1, by1))
    )

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def build_base(background_path, object_path, seed):
    """背景・대상・변형・위치를 한 번만 선택한다."""
    rng = np.random.default_rng(seed)

    background = read_image(background_path)

    if background is None:
        raise ValueError(f"배경 읽기 실패: {background_path}")

    background = fit_background(background, OUTPUT_SIZE)
    target = read_object(object_path)

    # 완전 투명하거나 유효 영역이 없는 대상 제외
    if not np.any(target[:, :, 3] > 127):
        raise ValueError(f"유효한 대상 영역 없음: {object_path}")

    long_side = (
        rng.uniform(*OBJECT_SCALE_RANGE)
        * min(OUTPUT_SIZE)
    )

    transformed = transform_object(
        target,
        rng,
        long_side,
        OBJECT_ANGLE_RANGE,
    )

    # floor를 주지 않으므로 이 단계에서는 가리지 않는다.
    composite, alpha = place_object(
        background,
        transformed,
        rng,
        margin=0.05,
        n_candidates=100,
        top_k=3,
    )

    if not np.any(alpha > 0.5):
        raise ValueError("변형 후 대상 영역이 사라졌습니다.")

    return background, composite, alpha, target


def apply_occlusion(background, composite, alpha, ratio, side):
    """같은 배치에서 대상의 한쪽부터 지정 면적만큼 제거한다.

    side:
        0=왼쪽, 1=오른쪽, 2=위, 3=아래

    대상 픽셀을 방향 순으로 정렬해 제거하므로,
    같은 side에서 가림 비율이 커지면 기존 가림을 포함한다.
    """
    keep = np.ones(alpha.shape, dtype=np.float32)

    # 보이는 대상 면적의 기준
    solid = alpha > 0.5
    ys, xs = np.nonzero(solid)

    hidden_count = int(round(len(xs) * ratio))

    if hidden_count:
        if side == 0:
            order = np.lexsort((ys, xs))
        elif side == 1:
            order = np.lexsort((ys, -xs))
        elif side == 2:
            order = np.lexsort((xs, ys))
        else:
            order = np.lexsort((xs, -ys))

        hidden = order[:hidden_count]
        keep[ys[hidden], xs[hidden]] = 0.0

        # 같은 방향의 투명 경계도 함께 제거
        if side == 0:
            boundary = xs[hidden].max()
            keep[:, :boundary] = 0
        elif side == 1:
            boundary = xs[hidden].min()
            keep[:, boundary + 1:] = 0
        elif side == 2:
            boundary = ys[hidden].max()
            keep[:boundary, :] = 0
        else:
            boundary = ys[hidden].min()
            keep[boundary + 1:, :] = 0

    visible_alpha = alpha * keep

    # 기존 합성 이미지에서 가릴 부분만 원래 배경으로 복원
    visible_composite = (
        background.astype(np.float32)
        + (
            composite.astype(np.float32)
            - background.astype(np.float32)
        ) * keep[:, :, None]
    )

    visible_composite = np.clip(
        np.rint(visible_composite), 0, 255
    ).astype(np.uint8)

    # 동일한 블렌딩 함수 사용
    image = blend_object(
        visible_composite,
        background,
        visible_alpha,
    )

    actual_ratio = (
        np.count_nonzero(solid & (keep == 0))
        / np.count_nonzero(solid)
    )

    return image, actual_ratio


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    backgrounds, objects = list_assets(ASSET_DIR)

    if not backgrounds or not objects:
        raise ValueError(
            "assets/backgrounds와 assets/objects에 이미지가 필요합니다."
        )

    checker = DifficultyCheck()
    rows = []
    results = {
        method: {ratio: [] for ratio in OCCLUSION_RATIOS}
        for method in (*METHODS, "orb/sift")
    }

    for index in range(NUM_SAMPLES):
        seed = START_SEED + index
        rng = np.random.default_rng(seed)

        background_path = backgrounds[
            int(rng.integers(len(backgrounds)))
        ]
        object_path = objects[
            int(rng.integers(len(objects)))
        ]

        # 이 표본의 모든 조건에서 같은 방향 사용
        side = int(rng.integers(4))

        background, composite, alpha, target = build_base(
            background_path, object_path, seed
        )

        # 모든 가림 조건에 동일한 평가 박스 사용
        answer_box = mask_box(alpha > 0.5)

        group_start = (seed // 10) * 10
        group_end = group_start + 9
        group_dir = OUTPUT_DIR / f"{group_start:04d}-{group_end:04d}"
        sample_dir = group_dir / f"seed_{seed:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_image(sample_dir / "target.png", target)

        for ratio in OCCLUSION_RATIOS:
            image, actual_ratio = apply_occlusion(
                background,
                composite,
                alpha,
                ratio,
                side,
            )

            # 정답 정보는 검수기에 전달하지 않는다.
            boxes_by_method = {
                method: checker.difficult_check(image, target, method=method)
                for method in METHODS
            }
            boxes_by_method["orb/sift"] = list(dict.fromkeys(
                box for method in METHODS for box in boxes_by_method[method]
            ))
            percent = round(ratio * 100)
            save_image(sample_dir / f"input_{percent:02d}.png", image)

            for method, boxes in boxes_by_method.items():
                best_iou = max(
                    (box_iou(answer_box, box) for box in boxes),
                    default=0.0,
                )

                correct = best_iou >= IOU_THRESHOLD

                if correct:
                    status = "correct"
                elif boxes:
                    status = "wrong_location"
                else:
                    status = "not_detected"

                results[method][ratio].append(correct)

                rows.append({
                    "seed": seed,
                    "method": method,
                    "background": background_path.name,
                    "object": object_path.name,
                    "side": side,
                    "requested_occlusion": ratio,
                    "actual_occlusion": actual_ratio,
                    "boxes": str(boxes),
                    "answer_box": str(answer_box),
                    "iou": best_iou,
                    "status": status,
                })

                preview = image.copy()

                # 빨간색: 가림 전 대상 전체의 정답 박스
                x1, y1, x2, y2 = answer_box
                cv2.rectangle(
                    preview,
                    (x1, y1), (x2 - 1, y2 - 1),
                    (0, 0, 255), 2,
                )

                # 초록색: 검수기가 추정한 박스
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(
                        preview,
                        (x1, y1), (x2 - 1, y2 - 1),
                        (0, 255, 0), 2,
                    )

                cv2.putText(
                    preview,
                    (
                        f"{method.upper()} seed={seed} hidden={actual_ratio:.1%} "
                        f"{status} IoU={best_iou:.3f}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                save_image(
                    sample_dir / f"result_{percent:02d}_{method}.png",
                    preview,
                )

                print(
                    f"[{index + 1}/{NUM_SAMPLES}] "
                    f"{method.upper()}, seed={seed}, 가림={actual_ratio:.1%}, "
                    f"상태={status}, IoU={best_iou:.3f}"
                )

        # 한 표본의 모든 조건을 처리한 뒤 창 표시
        if SHOW_WINDOW:
            cv2.namedWindow("Occlusion", cv2.WINDOW_NORMAL)
            cv2.imshow("Occlusion", preview)

            key = cv2.waitKey(WAIT_MS) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    csv_path = OUTPUT_DIR / "results.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== 조건별 결과 =====")

    for method, method_results in results.items():
        label = "ORB 또는 SIFT" if method == "orb/sift" else method.upper()
        print(f"\n[{label}]")
        baseline = method_results[0.0]

        for ratio, flags in method_results.items():
            total = len(flags)
            success = sum(flags)

            # 가림 없을 때 맞힌 표본 중 이 조건에서도 맞힌 개수
            baseline_success = sum(baseline)
            retained = sum(
                before and after
                for before, after in zip(baseline, flags)
            )

            print(
                f"가림 {ratio:.0%}: "
                f"정답 {success}/{total} "
                f"({success / total:.1%})"
            )

            if ratio > 0 and baseline_success:
                print(
                    f"  가림 전 정답 표본의 검출 유지율: "
                    f"{retained}/{baseline_success} "
                    f"({retained / baseline_success:.1%})"
                )

    print(f"\n결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
