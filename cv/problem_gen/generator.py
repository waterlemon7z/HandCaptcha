"""숨은그림찾기 문제 생성과 정답 판정. 정답 영역은 판정 전에는 이 클래스 밖으로 나가지 않는다."""
import secrets
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from .assets import list_assets, read_image, read_object
from .blending import pyramid_blend
from .compose import fit_background, floor_map, place_object, transform_object

OUTPUT_SIZE = (1280, 720)           # (w, h). 배경은 가운데를 잘라 이 크기로 맞춘다
NUM_OBJECTS = 3                     # 문제마다 합성하는 물체 수. 이 중 1개가 정답, 나머지는 방해용
OBJECT_SCALE_RANGE = (0.10, 0.16)   # 물체 긴 변(회전 전) / 배경 짧은 변
OBJECT_ANGLE_RANGE = (-45.0, 45.0)  # 물체 회전 각도(도)
PLACE_MARGIN = 0.05                 # 가장자리에서 띄울 비율. 손으로 짚기 어려운 화면 끝을 피한다
PLACE_CANDIDATES = 100              # 위치 후보 수. 자리는 비어 있고 주변은 복잡한 곳일수록 점수가 좋다
PLACE_TOP_K = 3                     # 점수 상위 몇 개 중에서 랜덤으로 고를지. 1이면 같은 배경·물체에서 항상 같은 자리
OCCLUDE_RANGE = (0.2, 0.45)         # 물체가 사진 속 물건 밑에 깔려 가려지는 비율 (최소, 최대)
BLEND_LEVELS = 4                    # 피라미드 단계 상한 (작은 물체는 자동으로 줄임)
BLEND_FEATHER_SIGMA = 2.0           # 블렌딩 마스크를 흐리는 정도(px). 0이면 alpha 그대로
HIT_IOU = 0.5                       # 입력 박스와 정답 박스의 IoU가 이 이상이면 정답


@dataclass
class _Answer:
    box: tuple          # 정답 물체의 보이는 부분 (x1, y1, x2, y2). x2, y2는 박스 바로 바깥(슬라이스 끝)
    mask: np.ndarray    # 정답 물체의 보이는 부분 마스크 (H×W, 0/255). 결과 화면 하이라이트용


def _iou(a, b):
    """두 박스 (x1, y1, x2, y2)의 IoU. 좌표 순서가 뒤집혀 들어와도 정렬해서 계산한다."""
    ax1, ax2 = sorted((float(a[0]), float(a[2])))
    ay1, ay2 = sorted((float(a[1]), float(a[3])))
    bx1, bx2 = sorted((float(b[0]), float(b[2])))
    by1, by2 = sorted((float(b[1]), float(b[3])))
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _hit(answer_box, box):
    return _iou(answer_box, box) >= HIT_IOU


def blend_object(composite, bg, alpha, feather_sigma=BLEND_FEATHER_SIGMA):
    """물체를 붙인 이미지를 배경과 라플라시안 피라미드로 섞는다."""
    # 단계가 너무 많으면 얇거나 작은 물체에 배경색이 번지므로 물체 두께에 맞춰 줄인다.
    # 회전한 연필처럼 박스는 커도 얇은 물체가 있으므로 박스 크기가 아니라 가장 두꺼운 곳의 폭을 쓴다
    mask = np.where(alpha > 0.5, 255, 0).astype(np.uint8)
    ox, oy, ow, oh = cv2.boundingRect(mask)
    thickness = 2 * cv2.distanceTransform(mask[oy:oy + oh, ox:ox + ow], cv2.DIST_L2, 5).max()
    levels = max(1, min(BLEND_LEVELS, int(np.log2(max(2, thickness) / 2))))
    # 마스크가 0인 곳은 배경 그대로이므로, 피라미드가 닿는 범위까지만 잘라서 섞는다 (전체를 섞으면 몇 배 느림)
    h, w = bg.shape[:2]
    pad = 8 * 2 ** levels + int(3 * feather_sigma)
    roi = (slice(max(0, oy - pad), min(h, oy + oh + pad)), slice(max(0, ox - pad), min(w, ox + ow + pad)))
    alpha = alpha[roi]
    # alpha 그대로면 가장 세밀한 단계에서 경계가 그대로 복사되므로, 마스크를 흐려 경계 픽셀도 섞이게 한다
    if feather_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
    out = bg.copy()
    out[roi] = pyramid_blend(composite[roi], bg[roi], alpha, levels)
    return out


class ProblemGenerator:
    def __init__(self, asset_dir="assets", validator=None, max_attempts=5):
        """
        validator(image, target) -> [(x1, y1, x2, y2), ...] | None
            정답을 모르는 자동 풀이 함수(ORB/SIFT 검수). 합성 이미지(H×W×3 BGR)와
            정답 물체 1개의 원본 이미지(BGRA, 방해용 물체는 넘기지 않음)를 받아
            찾은 영역들을 반환한다. alpha는 특징점 검출 마스크로 쓸 수 있다.
            그중 하나라도 정답이면 문제를 다시 만들고,
            max_attempts번 모두 풀리면 마지막 문제를 그대로 발급한다.
        """
        self._backgrounds, self._objects = list_assets(asset_dir)
        if len(self._objects) < NUM_OBJECTS:
            raise ValueError(f"물체 이미지가 {NUM_OBJECTS}개 이상 필요합니다: {len(self._objects)}개")
        self._validator = validator
        self._max_attempts = max_attempts
        self._answers = {}
        self._history = []

    def create(self, seed=None):
        """새 문제를 만든다. 반환: (problem_id, image, target).

        image: 물체 NUM_OBJECTS개를 숨긴 합성 이미지 (H×W×3 BGR)
        target: 찾아야 할 정답 물체의 원본 이미지 (BGRA). 사용자 화면에 "이것을 찾으세요"로 보여 주고, 검수 입력으로도 쓴다
        정답은 반환하지 않는다.
        """
        if seed is None:
            seed = secrets.randbits(32)
        # 재생성도 같은 rng를 이어 쓰므로 seed와 validator가 같으면 최종 문제도 같다
        rng = np.random.default_rng(seed)
        attempts = []
        for _ in range(self._max_attempts):
            image, target, answer = self._build(rng)
            if self._validator is None:
                break
            boxes = [tuple(int(round(float(v))) for v in b) for b in self._validator(image, target) or []]
            solved = any(_hit(answer.box, b) for b in boxes)
            attempts.append({"boxes": boxes, "solved": solved})
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
        return problem_id, image, target

    def solve(self, problem_id, box):
        """box(합성 이미지 기준 (x1, y1, x2, y2))가 정답 물체의 박스와 HIT_IOU 이상 겹치면 정답.

        봇이든 사람이든 같은 기준이다. 한 번 판정한 문제는 폐기하고, 맞든 틀리든 정답을 공개한다.
        """
        answer = self._answers.pop(problem_id, None)
        if answer is None:
            return {"correct": False, "answer_box": None, "answer_mask": None}
        return {"correct": _hit(answer.box, box), "answer_box": answer.box, "answer_mask": answer.mask}

    @property
    def history(self):
        """validator 검수 기록. validator가 없으면 비어 있다."""
        return list(self._history)

    def auto_solve_rate(self):
        """발급한 문제 중 validator가 첫 시도에 정답을 찾은 비율. 기록이 없으면 None."""
        if not self._history:
            return None
        return sum(h["attempts"][0]["solved"] for h in self._history) / len(self._history)

    def _build(self, rng):
        bg = read_image(self._backgrounds[rng.integers(len(self._backgrounds))])
        objects = [read_object(self._objects[i]) for i in rng.choice(len(self._objects), NUM_OBJECTS, replace=False)]
        image = fit_background(bg, OUTPUT_SIZE)
        floor = floor_map(image)

        occupied = np.zeros(image.shape[:2], bool)
        masks = []
        for obj in objects:
            long_side = rng.uniform(*OBJECT_SCALE_RANGE) * min(OUTPUT_SIZE)
            placed = transform_object(obj, rng, long_side, OBJECT_ANGLE_RANGE)
            composite, alpha = place_object(image, placed, rng, PLACE_MARGIN, PLACE_CANDIDATES, PLACE_TOP_K, occupied,
                                            floor, OCCLUDE_RANGE)
            image = blend_object(composite, image, alpha)
            mask = np.where(alpha > 0.5, 255, 0).astype(np.uint8)
            occupied |= mask > 0
            masks.append(mask)

        i = int(rng.integers(NUM_OBJECTS))
        x, y, w, h = cv2.boundingRect(masks[i])
        return image, objects[i], _Answer((x, y, x + w, y + h), masks[i])
