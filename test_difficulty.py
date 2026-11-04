from pathlib import Path

import cv2
import numpy as np

from cv.problem_gen import ProblemGenerator
from verification.difficulty_check import difficulty_check


# 한글 경로에서도 저장할 수 있도록 처리
def save_image(path, image):
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError(f"이미지 저장 실패: {path}")

    encoded.tofile(str(path))


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "test_results"
    result_dir = output_dir / "result"
    target_dir = output_dir / "target"
    result_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 검수 전 문제를 만들어 검출 성능을 직접 확인
    generator = ProblemGenerator(
        asset_dir=str(root / "assets")
    )

    total = 100
    detected_count = 0
    correct_count = 0

    for seed in range(total):
        problem_id, image, target = generator.create(seed=seed)

        # 1. 생성기 → 검수기 입력 확인
        assert image.dtype == np.uint8
        assert target.dtype == np.uint8

        # 2. 검수 실행
        boxes = difficulty_check(image, target)

        # 3. 검수기 → 생성기 반환 형식 확인
        assert isinstance(boxes, list), "반환값은 리스트여야 합니다."

        height, width = image.shape[:2]

        for box in boxes:
            assert len(box) == 4, "박스에는 좌표 4개가 필요합니다."
            assert all(isinstance(v, int) for v in box)

            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= width
            assert 0 <= y1 < y2 <= height

        # 현재 검수기는 최대 한 개의 박스를 반환
        submitted_box = boxes[0] if boxes else (0, 0, 0, 0)

        # 4. 생성기가 실제 정답과 비교
        answer = generator.solve(problem_id, submitted_box)

        detected_count += bool(boxes)
        correct_count += bool(answer["correct"])

        preview = image.copy()

        # 초록색: 검수기가 찾은 영역
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                preview,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (0, 255, 0),
                2,
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
            f"추정 좌표={boxes}, "
            f"정답 검출={answer['correct']}"
        )

    print("\n===== 테스트 결과 =====")
    print(f"위치 추정 결과 있음: {detected_count}/{total}")
    print(f"정답 검출 성공: {correct_count}/{total}")
    print(f"결과 이미지: {output_dir}")


if __name__ == "__main__":
    main()