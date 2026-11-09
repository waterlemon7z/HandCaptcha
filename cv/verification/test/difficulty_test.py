"""ORB/SIFT 개별 및 통합 검출 평가.

실행: python -m cv.verification.test.difficulty_test
"""

from pathlib import Path

import cv2
import numpy as np

from cv.problem_gen import ProblemGenerator
from cv.problem_gen.generator import _hit
from cv.verification.difficulty_check import DifficultyCheck


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("orb", "sift")
METHOD_COLORS = {"orb": (0, 255, 0), "sift": (255, 255, 0)}
TOTAL = 100


# 한글 경로에서도 저장할 수 있도록 처리
def save_image(path, image):
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError(f"이미지 저장 실패: {path}")

    encoded.tofile(str(path))


def main():
    root = ROOT
    output_dir = root / "test_results"
    result_dir = output_dir / "result"
    target_dir = output_dir / "target"
    result_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 검수 전 문제를 만들어 검출 성능을 직접 확인
    generator = ProblemGenerator(
        asset_dir=str(root / "assets")
    )

    checker = DifficultyCheck()
    total = TOTAL
    detected_count = {method: 0 for method in (*METHODS, "orb/sift")}
    correct_count = {method: 0 for method in (*METHODS, "orb/sift")}

    for seed in range(total):
        problem_id, image, target = generator.create(seed=seed)

        # 1. 생성기 → 검수기 입력 확인
        assert image.dtype == np.uint8
        assert target.dtype == np.uint8

        # 같은 문제에서 ORB와 SIFT를 각각 실행한다.
        boxes_by_method = {
            method: checker.difficult_check(image, target, method=method)
            for method in METHODS
        }
        # auto와 같은 합집합: 후보 중 하나라도 정답이면 성공이다.
        boxes_by_method["orb/sift"] = list(dict.fromkeys(
            box for method in METHODS for box in boxes_by_method[method]
        ))

        height, width = image.shape[:2]
        for boxes in boxes_by_method.values():
            assert isinstance(boxes, list), "반환값은 리스트여야 합니다."
            for box in boxes:
                assert len(box) == 4, "박스에는 좌표 4개가 필요합니다."
                assert all(isinstance(v, int) for v in box)
                x1, y1, x2, y2 = box
                assert 0 <= x1 < x2 <= width
                assert 0 <= y1 < y2 <= height

        # solve()는 정답을 폐기하므로 한 번만 호출한 뒤,
        # 공개된 정답으로 모든 후보를 생성기와 동일한 기준에서 평가한다.
        submitted_boxes = boxes_by_method["orb/sift"]
        answer = generator.solve(
            problem_id, submitted_boxes[0] if submitted_boxes else (0, 0, 0, 0)
        )
        correct_by_method = {}
        for method, boxes in boxes_by_method.items():
            correct = any(_hit(answer["answer_box"], box) for box in boxes)
            correct_by_method[method] = correct
            detected_count[method] += bool(boxes)
            correct_count[method] += correct

        preview = image.copy()

        # ORB: 초록색, SIFT: 하늘색
        for method in METHODS:
            for x1, y1, x2, y2 in boxes_by_method[method]:
                color = METHOD_COLORS[method]
                cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), color, 2)
                cv2.putText(
                    preview, method.upper(), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

        # 빨간색: 실제 정답 영역
        x1, y1, x2, y2 = answer["answer_box"]
        cv2.rectangle(
            preview,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 0, 255),
            2,
        )

        save_image(
            result_dir / f"result_{seed:02d}.png",
            preview,
        )
        save_image(
            target_dir / f"target_{seed:02d}.png",
            target,
        )

        print(
            f"[{seed + 1}/{total}] "
            f"추정 좌표={boxes_by_method}, "
            f"정답 검출={correct_by_method}"
        )

    print("\n===== 테스트 결과 =====")
    for method in detected_count:
        label = "ORB 또는 SIFT" if method == "orb/sift" else method.upper()
        print(f"{label}: 위치 추정 {detected_count[method]}/{total}, "
              f"정답 검출 {correct_count[method]}/{total}")
    print(f"결과 이미지: {output_dir}")


if __name__ == "__main__":
    main()
