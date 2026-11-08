import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


def select_device(requested="auto"):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("이 환경에서는 MPS를 사용할 수 없습니다.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("이 환경에서는 CUDA를 사용할 수 없습니다.")
    return torch.device(requested)


def to_rgb(image):
    """uint8 OpenCV 흑백/BGR/BGRA 배열을 RGB PIL 이미지로 변환한다."""
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("이미지는 비어 있지 않은 uint8 NumPy 배열이어야 합니다.")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.ndim == 3 and image.shape[2] in (3, 4):
        if image.shape[2] == 4:
            # 투명 영역의 임의 RGB 값이 참조 이미지에 노출되지 않게 한다.
            rgba = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    raise ValueError("흑백, BGR, BGRA 이미지만 지원합니다.")


class Owlv2Checker:
    def __init__(self, *, model_id=DEFAULT_MODEL, device="auto", threshold=0.1,
                 nms_threshold=0.3, max_boxes=1):
        if not 0 <= threshold <= 1 or not 0 <= nms_threshold <= 1:
            raise ValueError("threshold와 nms_threshold는 0~1이어야 합니다.")
        if max_boxes < 1:
            raise ValueError("max_boxes는 1 이상이어야 합니다.")
        self.device = select_device(device)
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes
        # 모델은 한 번만 로드한다. 모듈 import 시에는 로드하지 않는다.
        self.processor = Owlv2Processor.from_pretrained(model_id, backend="pil")
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image, target):
        """정답 정보 없이 대상의 xyxy 후보 박스를 점수 순으로 반환한다."""
        scene = to_rgb(image)
        reference = to_rgb(target)
        if target.ndim == 3 and target.shape[2] == 4:
            mask = target[:, :, 3]
            if not mask.any():
                return []
            x, y, w, h = cv2.boundingRect(mask)
            reference = reference.crop((x, y, x + w, y + h))

        inputs = self.processor(
            images=scene, query_images=reference, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model.image_guided_detection(**inputs)

        # 후처리의 NMS 루프는 CPU에서 수행해 GPU 동기화 반복을 피한다.
        cpu_outputs = SimpleNamespace(
            logits=outputs.logits.detach().cpu(),
            target_pred_boxes=outputs.target_pred_boxes.detach().cpu(),
        )
        results = self.processor.post_process_image_guided_detection(
            outputs=cpu_outputs,
            target_sizes=torch.tensor([[scene.height, scene.width]]),
            threshold=self.threshold,
            nms_threshold=self.nms_threshold,
        )
        if not results:
            return []
        result = results[0]
        # image-guided 후처리 scores는 시각화용 상대 점수이므로 확률로 표시하지 않는다.
        order = torch.argsort(result["scores"], descending=True)
        boxes = []
        for index in order.tolist():
            coords = result["boxes"][index].numpy()
            if not np.isfinite(coords).all():
                continue
            x1, y1 = np.floor(coords[:2]).astype(int)
            x2, y2 = np.ceil(coords[2:]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, scene.width)
            y1, y2 = np.clip([y1, y2], 0, scene.height)
            box = (int(x1), int(y1), int(x2), int(y2))
            if x2 > x1 and y2 > y1 and box not in boxes:
                boxes.append(box)
                if len(boxes) == self.max_boxes:
                    break
        return boxes


def save_image(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    encoded.tofile(str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description="생성한 캡챠에서 OWLv2 검출 성능 평가")
    parser.add_argument("--total", type=int, default=100, help="문제 수 (기본: 100)")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--max-boxes", type=int, default=1, help="문제당 후보 수 (기본: 1)")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "owlv2_results")
    args = parser.parse_args(argv)
    if args.total < 1 or args.start_seed < 0 or args.max_boxes < 1:
        parser.error("total/max-boxes는 1 이상, start-seed는 0 이상이어야 합니다.")
    if not 0 <= args.threshold <= 1 or not 0 <= args.nms_threshold <= 1:
        parser.error("threshold/nms-threshold는 0~1이어야 합니다.")

    from cv.problem_gen import ProblemGenerator
    from cv.problem_gen.generator import _hit

    # 자동 재생성 없이 ORB 실험과 동일한 seed의 문제를 평가한다.
    generator = ProblemGenerator(asset_dir=str(args.asset_dir))
    checker = Owlv2Checker(
        model_id=args.model, device=args.device, threshold=args.threshold,
        nms_threshold=args.nms_threshold, max_boxes=args.max_boxes,
    )
    print(f"장치: {checker.device}, 모델: {args.model}, 후보 수: {args.max_boxes}", flush=True)
    for name in ("input", "result", "target"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({**vars(args), "actual_device": str(checker.device)},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    rows = []
    for index in range(args.total):
        seed = args.start_seed + index
        problem_id, image, target = generator.create(seed=seed)
        started = perf_counter()
        boxes = checker.detect(image, target)
        elapsed = perf_counter() - started
        # solve는 한 번만 호출하고, 공개된 정답으로 모든 후보를 평가한다.
        answer = generator.solve(problem_id, boxes[0] if boxes else (0, 0, 0, 0))
        correct = any(_hit(answer["answer_box"], box) for box in boxes)
        status = "correct" if correct else "wrong_location" if boxes else "not_detected"
        preview = image.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
            cv2.putText(preview, "OWLv2", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        x1, y1, x2, y2 = answer["answer_box"]
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        save_image(args.output_dir / "input" / f"input_{seed:04d}.png", image)
        save_image(args.output_dir / "result" / f"result_{seed:04d}.png", preview)
        save_image(args.output_dir / "target" / f"target_{seed:04d}.png", target)
        rows.append({"seed": seed, "boxes": json.dumps(boxes),
                     "answer_box": json.dumps(answer["answer_box"]),
                     "status": status, "seconds": elapsed})
        # 각 표본 후 기록해 긴 실험을 중단해도 완료된 결과를 보존한다.
        with (args.output_dir / "results.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{index + 1}/{args.total}] seed={seed}, 박스={boxes}, "
              f"상태={status}, 시간={elapsed:.3f}s", flush=True)

    detected = sum(row["status"] != "not_detected" for row in rows)
    correct_count = sum(row["status"] == "correct" for row in rows)
    print(f"\n위치 추정: {detected}/{args.total}")
    print(f"정답 검출: {correct_count}/{args.total} ({correct_count / args.total:.1%})")
    print(f"평균 시간 (첫 추론 포함, 모델 로딩 제외): "
          f"{sum(row['seconds'] for row in rows) / args.total:.3f}s")
    print(f"결과 저장: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
