"""문제 생성과 정답 판정. 정답 영역은 이 클래스 밖으로 판정 전에는 나가지 않는다."""
import secrets
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from .assets import list_assets, read_image, read_object
from .blending import pyramid_blend
from .compose import fit_background, place_object, random_affine, warp

OUTPUT_SIZE = (640, 480)            # (w, h). 배경은 가운데를 잘라 이 크기로 맞춘다
OBJECT_SCALE_RANGE = (0.15, 0.25)   # 물체 긴 변 / 배경 짧은 변
# 아래 회전·확대 범위에서 물체가 프레임 밖으로 잘리지 않으려면 가장자리 여백이 0.13 이상 필요하다
PLACE_MARGIN = 0.14
ANGLE_RANGE = (3.0, 6.0)            # 오른쪽 이미지 회전 각도(도), 부호는 랜덤
EXTRA_SCALE_RANGE = (1.0, 1.05)     # 빈 모서리를 없애는 최소 배율에 추가로 곱하는 배율
BLEND_LEVELS = 4                    # 피라미드 단계 상한 (작은 물체는 자동으로 줄임)
HIT_TOLERANCE_PX = 8                # 손끝 오차 허용: 판정할 때 정답 마스크를 넓히는 픽셀 수


@dataclass
class _Answer:
    mask: np.ndarray      # 오른쪽 이미지 기준 정답 마스크 (H×W, 0/255)
    box: tuple            # mask를 감싸는 사각형 (x, y, w, h)
    hit_mask: np.ndarray  # 판정용으로 HIT_TOLERANCE_PX만큼 넓힌 마스크
    affine: np.ndarray    # 왼쪽 → 오른쪽 변환 행렬. "왼쪽을 짚어도 인정"할지 정해지면 쓴다


def _to_pixel(point):
    return int(round(float(point[0]))), int(round(float(point[1])))


def _hit(mask, point):
    x, y = _to_pixel(point)
    h, w = mask.shape
    return 0 <= x < w and 0 <= y < h and bool(mask[y, x] > 0)


class ProblemGenerator:
    def __init__(self, asset_dir="assets", validator=None, max_attempts=5):
        """
        validator(left_img, right_img) -> (x, y) | None
            정답을 모르는 자동 풀이 함수(ORB/SIFT 검수). 오른쪽 이미지 기준 예측 위치를 반환한다.
            예측이 정답 영역 안이면 문제를 다시 만들고, max_attempts번 모두 풀리면 마지막 문제를 그대로 발급한다.
        """
        self._backgrounds, self._objects = list_assets(asset_dir)
        self._validator = validator
        self._max_attempts = max_attempts
        self._answers = {}
        self._history = []

    def create(self, seed=None):
        """새 문제를 만든다. 반환: (problem_id, left_img, right_img). 정답은 반환하지 않는다."""
        if seed is None:
            seed = secrets.randbits(32)
        # 재생성도 같은 rng를 이어 쓰므로 seed와 validator가 같으면 최종 문제도 같다
        rng = np.random.default_rng(seed)
        attempts = []
        for _ in range(self._max_attempts):
            left, right, answer = self._build(rng)
            if self._validator is None:
                break
            predicted = self._validator(left, right)
            solved = predicted is not None and _hit(answer.hit_mask, predicted)
            attempts.append({"predicted": None if predicted is None else _to_pixel(predicted), "solved": solved})
            if not solved:
                break

        problem_id = uuid.uuid4().hex
        self._answers[problem_id] = answer
        if self._validator is not None:
            self._history.append({
                "problem_id": problem_id,
                "seed": seed,
                "attempts": attempts,
                "cap_reached": attempts[-1]["solved"],
            })
        return problem_id, left, right

    def solve(self, problem_id, point):
        """point(오른쪽 이미지 기준 (x, y))가 정답인지 판정한다. 한 번 판정한 문제는 폐기한다."""
        answer = self._answers.pop(problem_id, None)
        if answer is None:
            return {"correct": False, "answer_mask": None, "answer_box": None}
        return {"correct": _hit(answer.hit_mask, point), "answer_mask": answer.mask, "answer_box": answer.box}

    @property
    def history(self):
        """validator 검수 기록. validator가 없으면 비어 있다."""
        return list(self._history)

    def auto_solve_rate(self):
        """발급한 문제 중 validator가 첫 시도에 푼 비율. 기록이 없으면 None."""
        if not self._history:
            return None
        return sum(h["attempts"][0]["solved"] for h in self._history) / len(self._history)

    def _build(self, rng):
        bg = read_image(self._backgrounds[rng.integers(len(self._backgrounds))])
        obj = read_object(self._objects[rng.integers(len(self._objects))])
        left = fit_background(bg, OUTPUT_SIZE)

        composite, alpha = place_object(left, obj, rng, OBJECT_SCALE_RANGE, PLACE_MARGIN)
        mask_left = np.where(alpha > 0.5, 255, 0).astype(np.uint8)
        # 단계가 너무 많으면 작은 물체에 배경색이 번지므로 물체 크기에 맞춰 줄인다
        _, _, ow, oh = cv2.boundingRect(mask_left)
        levels = max(1, min(BLEND_LEVELS, int(np.log2(max(2, min(ow, oh)) / 2))))
        blended = pyramid_blend(composite, left, alpha, levels)

        M = random_affine(*OUTPUT_SIZE, rng, ANGLE_RANGE, EXTRA_SCALE_RANGE)
        right, mask = warp(blended, mask_left, M)
        k = 2 * HIT_TOLERANCE_PX + 1
        hit_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return left, right, _Answer(mask, cv2.boundingRect(mask), hit_mask, M)
