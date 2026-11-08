"""캡챠 실패 분석 및 크기·가림 비교 실험.

프로젝트 루트에서 실행:
    python -m cv.verification.test.verification_experiment failures
    python -m cv.verification.test.verification_experiment compare --samples 100
    python -m cv.verification.test.verification_experiment summarize

결과는 verification_results/failures, verification_results/conditions에 저장한다.
모델 import만으로는 가중치를 로드하지 않으며, 실패 분석/요약은 추론을 실행하지 않는다.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

from cv.problem_gen.assets import list_assets, read_image, read_object
from cv.problem_gen.compose import fit_background, place_object, transform_object
from cv.problem_gen.generator import _hit, _iou, HIT_IOU, HIT_PAD_PX, blend_object
from cv.verification.orb_sift import OrbSiftChecker
from cv.verification.owlv2 import DEFAULT_MODEL, Owlv2Checker, save_image


ROOT = Path(__file__).resolve().parents[3]
LABELS = {
    "other_region": "Other region (no overlap)",
    "partial_target": "Partial target box",
    "oversized": "Oversized box",
    "localization_error": "Overlapping, poorly localized",
    "not_detected": "No detection",
}


SCALES = {"small": 0.08, "medium": 0.16, "large": 0.24}
OCCLUSIONS = (0.0, 0.25, 0.40)
METHODS = ("orb", "sift", "orb_sift", "owlv2")
COLORS = {"orb": "#8b5cf6", "sift": "#0891b2", "orb_sift": "#16a34a", "owlv2": "#e87916"}
DISPLAY_NAMES = {"orb": "ORB", "sift": "SIFT", "orb_sift": "ORB + SIFT", "owlv2": "OWLv2"}
OUTPUT_SIZE = (1280, 720)


def geometry(predicted, answer):
    px1, py1, px2, py2 = predicted
    ax1, ay1, ax2, ay2 = answer
    area = max(0, px2 - px1) * max(0, py2 - py1)
    answer_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    intersection = max(0, min(px2, ax2) - max(px1, ax1)) * max(0, min(py2, ay2) - max(py1, ay1))
    union = area + answer_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "target_coverage": intersection / answer_area if answer_area else 0.0,
        "prediction_overlap": intersection / area if area else 0.0,
        "area_ratio": area / answer_area if answer_area else 0.0,
        "prediction_area": area,
    }


def classify(predicted, answer, image_size):
    stats = geometry(predicted, answer)
    width, height = image_size
    if (stats["prediction_area"] >= 0.2 * width * height
            or (stats["area_ratio"] >= 4 and stats["target_coverage"] >= 0.5)):
        category = "oversized"
    elif stats["prediction_overlap"] >= 0.5 and stats["area_ratio"] < 0.6:
        category = "partial_target"
    elif stats["iou"] > 0:
        category = "localization_error"
    else:
        category = "other_region"
    return category, stats


def find_image(directory, prefix, seed):
    for digits in (4, 2):
        path = directory / f"{prefix}_{seed:0{digits}d}.png"
        if path.exists():
            return path
    raise FileNotFoundError(f"표본 이미지 없음: {directory}, seed={seed}")


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    reviews = {}
    review_path = output / "visual_review.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as file:
            reviews = {int(row["seed"]): row for row in csv.DictReader(file)}
    with (source / "results.csv").open(encoding="utf-8-sig", newline="") as file:
        original = list(csv.DictReader(file))
    rows = []
    cards = []
    for row in original:
        if row["status"] == "correct":
            continue
        seed = int(row["seed"])
        boxes = json.loads(row["boxes"])
        answer = json.loads(row["answer_box"])
        with Image.open(find_image(source / "input", "input", seed)) as image:
            image = image.convert("RGB")
        if boxes:
            # 여러 후보인 경우 정답과 IoU가 가장 큰 실패 후보를 분석한다.
            best = max(boxes, key=lambda box: geometry(box, answer)["iou"])
            category, stats = classify(best, answer, image.size)
        else:
            category, stats = "not_detected", dict.fromkeys(
                ("iou", "target_coverage", "prediction_overlap", "area_ratio", "prediction_area"), 0.0
            )
        review = reviews.get(seed, {})
        rows.append({**row, "category": category, **stats,
                     "classification_basis": "box_geometry",
                     "visual_review": review.get("visual_review", "pending"),
                     "visual_note": review.get("visual_note", "")})
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(box, outline="lime", width=4)
        draw.rectangle(answer, outline="red", width=4)
        with Image.open(find_image(source / "target", "target", seed)) as target:
            target = target.convert("RGBA")
            bbox = target.getbbox()
            if bbox:
                target = target.crop(bbox)
            target.thumbnail((84, 65))
            tile = Image.new("RGB", (480, 350), "white")
            tile.paste(ImageOps.contain(image, (480, 270)), (0, 0))
            tile.paste(target, (385, 278), target)
        caption = ImageDraw.Draw(tile)
        caption.text((8, 277), f"seed={seed}  {category}", fill="black")
        caption.text((8, 296), f"IoU={stats['iou']:.3f} area/GT={stats['area_ratio']:.2f}", fill="black")
        caption.text((8, 315), "green=prediction; red=answer; query ->", fill="black")
        cards.append(tile)
    if not rows:
        raise ValueError("분석할 실패 표본이 없습니다.")
    with (output / "failure_details.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for page, start in enumerate(range(0, len(cards), 12), 1):
        sheet = Image.new("RGB", (480 * 3, 350 * 4), "#eeeeee")
        for index, card in enumerate(cards[start:start + 12]):
            sheet.paste(card, ((index % 3) * 480, (index // 3) * 350))
        sheet.save(output / f"review_{page:02d}.jpg", quality=92)
    counts = Counter(row["category"] for row in rows)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.barh([LABELS[key] for key in counts], list(counts.values()), color="#478fdf")
    for index, value in enumerate(counts.values()):
        ax.text(value + 0.3, index, str(value), va="center")
    ax.set_xlabel("Failed samples")
    ax.set_title(f"OWLv2 failure categories ({len(rows)}/{len(original)} samples)")
    ax.set_xlim(0, max(counts.values()) * 1.15)
    fig.savefig(output / "failure_categories.png", dpi=180)
    plt.close(fig)
    report = ["# OWLv2 실패 분석", "", f"전체 {len(original)}개 중 실패 {len(rows)}개.", "",
              "자동 분류는 박스의 기하 관계를 설명한다. 다른 물체를 인식했는지, 배경을 인식했는지는 좌표만으로 확정하지 않는다.", "",
              "|분류|개수|실패 중 비율|", "|---|---:|---:|"]
    for key, count in counts.items():
        report.append(f"|{key}|{count}|{count / len(rows):.1%}|")
    report += ["", "분류 규칙(순서대로 적용):",
               "- oversized: 예측 면적이 장면의 20% 이상, 또는 정답의 4배 이상이면서 정답의 50% 이상을 포함.",
               "- partial_target: 예측의 50% 이상이 정답 안에 있고 예측 면적이 정답의 60% 미만.",
               "- localization_error: 위 조건 외 정답과 일부 겹침.",
               "- other_region: 정답과 겹치지 않음. 다른 객체/배경은 별도 육안 확인 필요.",
               "", "review_*.jpg: 초록=예측, 빨강=정답, 오른쪽 아래=참조 대상.",
               "failure_details.csv의 visual_review 열은 검토 기록이 없으면 pending이다.",
               f"이미지 검토 완료: {sum(row['visual_review'] == 'reviewed_geometry' for row in rows)}/{len(rows)}.",
               "visual_review.csv는 검토 기록이며 재실행할 때 자동으로 읽는다."]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(dict(counts))
    return rows


def mask_box(mask):
    """마스크를 감싸는 xyxy 박스."""
    x, y, w, h = cv2.boundingRect(
        mask.astype(np.uint8)
    )
    return x, y, x + w, y + h


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


def trimmed(image):
    x, y, width, height = cv2.boundingRect(image[:, :, 3])
    if not width or not height:
        raise ValueError("대상이 완전히 투명합니다.")
    return image[y:y + height, x:x + width].copy()


def place_at(background, transformed, center):
    height, width = transformed.shape[:2]
    x = round(center[0] - width / 2)
    y = round(center[1] - height / 2)
    if x < 0 or y < 0 or x + width > background.shape[1] or y + height > background.shape[0]:
        raise ValueError("대상이 이미지 밖으로 나갑니다.")
    alpha = np.zeros(background.shape[:2], np.float32)
    alpha[y:y + height, x:x + width] = transformed[:, :, 3] / 255.0
    composite = background.astype(np.float32)
    a = alpha[y:y + height, x:x + width, None]
    composite[y:y + height, x:x + width] = (
        transformed[:, :, :3] * a + composite[y:y + height, x:x + width] * (1 - a)
    )
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8), alpha


def build_conditions(background, target, seed):
    """최대 크기에서 위치를 한 번 선택하고 모든 크기의 중심과 회전을 고정한다."""
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-45, 45))
    side = int(rng.integers(4))
    cropped = trimmed(target)
    transformed = {
        name: trimmed(transform_object(
            cropped, np.random.default_rng(seed), scale * min(OUTPUT_SIZE), (angle, angle)
        ))
        for name, scale in SCALES.items()
    }
    _, largest_alpha = place_object(
        background, transformed["large"], rng, margin=0.05, n_candidates=100, top_k=3
    )
    x, y, width, height = cv2.boundingRect((largest_alpha > 0).astype(np.uint8))
    center = (x + width / 2, y + height / 2)
    for size, obj in transformed.items():
        composite, alpha = place_at(background, obj, center)
        full_box = mask_box(alpha > 0.5)
        for ratio in OCCLUSIONS:
            image, actual = apply_occlusion(background, composite, alpha, ratio, side)
            yield size, ratio, image, full_box, {
                "angle": angle, "side": side, "center_x": center[0], "center_y": center[1],
                "requested_long_side_px": SCALES[size] * min(OUTPUT_SIZE),
                "actual_occlusion": actual,
            }


def summarize(rows, output):
    summaries = []
    for size in SCALES:
        for ratio in OCCLUSIONS:
            for method in METHODS:
                group = [r for r in rows if r["size"] == size and r["occlusion"] == ratio and r["method"] == method]
                if not group:
                    continue
                successes = sum(r["correct"] for r in group)
                accuracy = successes / len(group)
                # Wilson 95% interval; uncertainty is across seeds, not 9 pooled conditions.
                z = 1.96
                denominator = 1 + z * z / len(group)
                center = (accuracy + z * z / (2 * len(group))) / denominator
                radius = z * sqrt(accuracy * (1 - accuracy) / len(group)
                                  + z * z / (4 * len(group) ** 2)) / denominator
                baseline = {r["seed"]: r["correct"] for r in rows
                            if r["size"] == size and r["occlusion"] == 0 and r["method"] == method}
                baseline_success = sum(baseline.get(r["seed"], False) for r in group)
                retained = sum(r["correct"] and baseline.get(r["seed"], False) for r in group)
                summaries.append({
                    "size": size, "occlusion": ratio, "method": method, "total": len(group),
                    "correct": successes, "accuracy": accuracy,
                    "accuracy_ci95_low": max(0.0, center - radius),
                    "accuracy_ci95_high": min(1.0, center + radius),
                    "baseline_correct": baseline_success,
                    "retained_correct": retained,
                    "retention_rate": retained / baseline_success if baseline_success else "",
                    "strict_iou50_accuracy": sum(r["strict_iou50"] for r in group) / len(group),
                    "wrong_location": sum(r["status"] == "wrong_location" for r in group),
                    "not_detected": sum(r["status"] == "not_detected" for r in group),
                    "mean_seconds": sum(r["seconds"] for r in group) / len(group),
                })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    for ax, size in zip(axes, SCALES):
        for method in METHODS:
            points = [r for r in summaries if r["size"] == size and r["method"] == method]
            errors = [[max(0, r["accuracy"] - r["accuracy_ci95_low"]) * 100 for r in points],
                      [max(0, r["accuracy_ci95_high"] - r["accuracy"]) * 100 for r in points]]
            ax.errorbar([r["occlusion"] * 100 for r in points], [r["accuracy"] * 100 for r in points],
                        yerr=errors, marker="s" if method == "orb_sift" else "o", capsize=3,
                        linestyle="--" if method == "orb_sift" else "-",
                        label=DISPLAY_NAMES[method], color=COLORS[method])
        ax.set_title(f"{size.title()}: {SCALES[size] * 720:.0f}px before rotation")
        ax.set_xlabel("Occluded foreground pixels (%)")
        ax.set_xticks([0, 25, 40])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Correct detections (%)")
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "accuracy_by_conditions.png", dpi=180)
    plt.close(fig)
    report = ["# 크기·가림 통제 실험", "",
              "각 seed에서 배경·대상·회전·중심 위치·가림 방향을 고정했다. 크기별로 가림 전 전체 대상 박스를 정답으로 사용했다.",
              "배경 위에 대상 하나만 합성하는 통제 실험이다. 방해 객체 2개가 추가되고 자연 가림이 있는 기존 생성기 결과(53%)와 직접 비교하면 안 된다.",
              "크기는 투명 여백을 제거한 대상의 회전 전 긴 변 기준 57.6 / 115.2 / 172.8px이다.",
              "가림은 배경 복원으로 대상 픽셀의 0/25/40%를 한 방향에서 제거한다. 자연스러운 가림의 완전한 재현은 아니다.",
              f"주 판정: 생성기의 _hit (IoU ≥ {HIT_IOU}, 정답 박스 {HIT_PAD_PX}px 확장도 허용). 작은 대상에 확장 판정이 유리할 수 있어 확장 없는 IoU≥0.5도 CSV에 기록했다.",
              "ORB·SIFT·OWLv2는 각각 후보 최대 1개. ORB/SIFT 합집합은 최대 2개이므로 별도 앙상블 결과로 해석한다.",
              "", "|크기|가림|ORB|SIFT|ORB/SIFT|OWLv2|", "|---|---:|---:|---:|---:|---:|"]
    for size in SCALES:
        for ratio in OCCLUSIONS:
            group = {r["method"]: r for r in summaries if r["size"] == size and r["occlusion"] == ratio}
            if len(group) == len(METHODS):
                values = [f"{group[m]['correct']}/{group[m]['total']} ({group[m]['accuracy']:.0%})" for m in METHODS]
                report.append(f"|{size}|{ratio:.0%}|" + "|".join(values) + "|")
    report += ["", "시간은 전처리·추론·CPU 후처리를 포함하고 모델 로딩은 제외한다. 첫 추론도 포함한다.",
               "보라=ORB, 하늘색=SIFT, 주황=OWLv2, 빨강=정답. previews는 처음 지정한 seed 수만 저장한다.",
               "같은 seed의 9조건은 서로 독립 표본이 아니다. 표본 수는 조건별 seed 수로 해석해야 한다.",
               "같은 배경·물체 자산을 여러 seed에서 재사용한다. 새로운 배경/물체에 대한 일반화 성능을 의미하지 않는다.",
               "그래프 오차막대는 조건별 seed를 기준으로 한 Wilson 95% 신뢰구간이다.",
               "summary.csv의 retention_rate는 가림 없는 조건에서 맞힌 seed 중 해당 가림 조건에서도 맞힌 비율이다."]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summaries


def run_conditions(args, parser):
    if args.samples < 1 or args.start_seed < 0 or args.preview_samples < 0:
        parser.error("samples는 1 이상, start-seed/preview-samples는 0 이상이어야 합니다.")
    if args.summarize_only:
        with (args.output_dir / "results.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            row["seed"] = int(row["seed"])
            for key in ("occlusion", "seconds"):
                row[key] = float(row[key])
            for key in ("correct", "strict_iou50"):
                row[key] = row[key] == "True"
        if not rows:
            parser.error("요약할 평가 행이 없습니다.")
        summarize(rows, args.output_dir)
        return
    backgrounds, objects = list_assets(args.asset_dir)
    if not backgrounds or not objects:
        parser.error("배경과 대상 이미지가 필요합니다.")
    if (args.output_dir / "results.csv").exists():
        parser.error("이 출력 폴더에 결과가 있습니다. 새 output-dir을 지정하세요.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    orb_sift = OrbSiftChecker()
    owl = Owlv2Checker(model_id=args.model, device=args.device, threshold=args.threshold, max_boxes=1)
    manifest = {
        **vars(args), "actual_device": str(owl.device), "scales": SCALES,
        "occlusions": OCCLUSIONS, "output_size": OUTPUT_SIZE,
        "hit_iou": HIT_IOU, "hit_pad_px": HIT_PAD_PX,
        "assets": {str(path.relative_to(args.asset_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (*backgrounds, *objects)},
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    rows = []
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = None
        for index in range(args.samples):
            seed = args.start_seed + index
            rng = np.random.default_rng(seed)
            background_path = backgrounds[int(rng.integers(len(backgrounds)))]
            object_path = objects[int(rng.integers(len(objects)))]
            background = fit_background(read_image(background_path), OUTPUT_SIZE)
            target = read_object(object_path)
            preview_dir = args.output_dir / "previews" / f"seed_{seed:04d}"
            if index < args.preview_samples:
                preview_dir.mkdir(parents=True, exist_ok=True)
                save_image(preview_dir / "target.png", target)
            for size, ratio, image, answer, metadata in build_conditions(background, target, seed):
                predictions, seconds = {}, {}
                for method in ("orb", "sift", "owlv2"):
                    cv2.setRNGSeed(seed)
                    started = perf_counter()
                    predictions[method] = (owl.detect(image, target) if method == "owlv2"
                                           else orb_sift.detect(image, target, method=method))
                    seconds[method] = perf_counter() - started
                predictions["orb_sift"] = list(dict.fromkeys(predictions["orb"] + predictions["sift"]))
                seconds["orb_sift"] = seconds["orb"] + seconds["sift"]
                for method in METHODS:
                    boxes = predictions[method]
                    correct = any(_hit(answer, box) for box in boxes)
                    iou = max((_iou(answer, box) for box in boxes), default=0.0)
                    status = "correct" if correct else "wrong_location" if boxes else "not_detected"
                    row = {"seed": seed, "background": background_path.name, "object": object_path.name,
                           "size": size, "occlusion": ratio, **metadata, "method": method,
                           "boxes": json.dumps(boxes), "answer_box": json.dumps(answer),
                           "correct": correct, "strict_iou50": iou >= 0.5,
                           "iou": iou, "status": status, "seconds": seconds[method]}
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    rows.append(row)
                file.flush()
                if index < args.preview_samples:
                    preview = image.copy()
                    for method, color in (("orb", (255, 0, 255)), ("sift", (255, 255, 0)), ("owlv2", (0, 140, 255))):
                        for x1, y1, x2, y2 in predictions[method]:
                            cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), color, 2)
                            cv2.putText(preview, method, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    x1, y1, x2, y2 = answer
                    cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
                    save_image(preview_dir / f"{size}_{ratio:.0%}.png", preview)
            print(f"[{index + 1}/{args.samples}] seed={seed} 완료 "
                  f"({(index + 1) * 9} images)", flush=True)
    summaries = summarize(rows, args.output_dir)
    print(f"완료: {len(rows)} 평가 행, {len(summaries)} 조건별 요약. {args.output_dir}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    failures = commands.add_parser("failures", help="기존 OWLv2 결과의 실패 분류")
    failures.add_argument("--source", type=Path, default=ROOT / "owlv2_results")
    failures.add_argument("--output-dir", type=Path, default=ROOT / "verification_results/failures")
    compare = commands.add_parser("compare", help="크기·가림 조건별 검출 성능 비교")
    compare.add_argument("--samples", type=int, default=100, help="조건별 동일 seed 수")
    compare.add_argument("--start-seed", type=int, default=0)
    compare.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    compare.add_argument("--threshold", type=float, default=0.1)
    compare.add_argument("--model", default=DEFAULT_MODEL)
    compare.add_argument("--asset-dir", type=Path, default=ROOT / "assets")
    compare.add_argument("--output-dir", type=Path, default=ROOT / "verification_results/conditions")
    compare.add_argument("--preview-samples", type=int, default=5)
    compare.set_defaults(summarize_only=False)
    summary = commands.add_parser("summarize", help="기존 CSV에서 표와 그래프만 다시 생성")
    summary.add_argument("--output-dir", type=Path, default=ROOT / "verification_results/conditions")
    summary.set_defaults(summarize_only=True, samples=100, start_seed=0, preview_samples=0)
    args = parser.parse_args(argv)
    if args.command == "failures":
        analyze(args.source, args.output_dir)
    else:
        run_conditions(args, parser)


if __name__ == "__main__":
    main()
