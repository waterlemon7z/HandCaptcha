"""숨은그림찾기 문제 생성과 정답 판정. 정답 영역은 문제가 끝나기 전에는 이 클래스 밖으로 나가지 않는다."""
import secrets
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from .assets import list_assets, read_image, read_object
from .blending import pyramid_blend
from .compose import fit_background, place_object, transform_object

OUTPUT_SIZE = (1280, 720)           # (w, h). 배경은 가운데를 잘라 이 크기로 맞춘다
NUM_OBJECTS = 3                     # 문제마다 숨기는 물체 수
OBJECT_SCALE_RANGE = (0.10, 0.16)   # 물체 긴 변(회전 전) / 배경 짧은 변
OBJECT_ANGLE_RANGE = (-45.0, 45.0)  # 물체 회전 각도(도)
PLACE_MARGIN = 0.05                 # 가장자리에서 띄울 비율. 손으로 짚기 어려운 화면 끝을 피한다
PLACE_CANDIDATES = 100              # 위치 후보 수. 자리는 비어 있고 주변은 복잡한 곳일수록 점수가 좋다
PLACE_TOP_K = 3                     # 점수 상위 몇 개 중에서 랜덤으로 고를지. 1이면 같은 배경·물체에서 항상 같은 자리
BLEND_LEVELS = 4                    # 피라미드 단계 상한 (작은 물체는 자동으로 줄임)
BLEND_FEATHER_SIGMA = 2.0           # 블렌딩 마스크를 흐리는 정도(px). 0이면 alpha 그대로
HIT_BOX_PX = 17                     # 판정 상자 한 변(px). 짚은 점(검수 박스는 그 중심)을 가운데에 둔다
HIT_RATIO = 0.3                     # 판정 상자 중 정답 마스크 안에 든 비율이 이 이상이면 정답


@dataclass
class _Answer:
    masks: list                               # 물체별 정답 마스크 (H×W, 0/255)
    boxes: list                               # 물체별 (x1, y1, x2, y2). x2, y2는 박스 바로 바깥(슬라이스 끝)
    found: set = field(default_factory=set)   # 지금까지 찾은 물체 번호


def _to_pixel(point):
    return int(round(float(point[0]))), int(round(float(point[1])))


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _hit(mask, point):
    """point를 가운데로 한 HIT_BOX_PX 상자 중 HIT_RATIO 이상이 마스크 안이면 정답. 이미지 밖은 마스크 밖으로 센다."""
    x, y = _to_pixel(point)
    r = HIT_BOX_PX // 2
    region = mask[max(0, y - r):max(0, y + r + 1), max(0, x - r):max(0, x + r + 1)]
    return bool(np.count_nonzero(region) >= HIT_RATIO * HIT_BOX_PX ** 2)


def _match(masks, point, skip=()):
    """point가 맞힌 물체 번호. skip에 든 물체는 보지 않는다. 없으면 None."""
    for i, mask in enumerate(masks):
        if i not in skip and _hit(mask, point):
            return i
    return None


def _count_found(masks, boxes):
    """검수 박스들로 찾은 물체 수. 박스가 어떤 물체인지는 모르므로 중심이 맞힌 물체를 하나씩 센다."""
    found = set()
    for box in boxes:
        i = _match(masks, _box_center(box), found)
        if i is not None:
            found.add(i)
    return len(found)


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
        validator(image, hints) -> [(x1, y1, x2, y2), ...] | None
            정답을 모르는 자동 풀이 함수(ORB/SIFT 검수). 합성 이미지와 물체 원본 이미지들을 받아
            찾은 영역들을 반환한다. 어떤 박스가 어떤 물체인지는 몰라도 된다.
            물체를 모두 찾으면 문제를 다시 만들고, max_attempts번 모두 풀리면 마지막 문제를 그대로 발급한다.
        """
        self._backgrounds, self._objects = list_assets(asset_dir)
        if len(self._objects) < NUM_OBJECTS:
            raise ValueError(f"물체 이미지가 {NUM_OBJECTS}개 이상 필요합니다: {len(self._objects)}개")
        self._validator = validator
        self._max_attempts = max_attempts
        self._answers = {}
        self._history = []

    def create(self, seed=None):
        """새 문제를 만든다. 반환: (problem_id, image, hints).

        image: 물체를 숨긴 합성 이미지 (H×W×3 BGR)
        hints: 숨긴 물체의 원본 이미지 리스트 (BGRA). 사용자에게 "이것들을 찾으세요"로 보여 준다
        정답은 반환하지 않는다.
        """
        if seed is None:
            seed = secrets.randbits(32)
        # 재생성도 같은 rng를 이어 쓰므로 seed와 validator가 같으면 최종 문제도 같다
        rng = np.random.default_rng(seed)
        attempts = []
        for _ in range(self._max_attempts):
            image, hints, answer = self._build(rng)
            if self._validator is None:
                break
            boxes = [tuple(int(round(float(v))) for v in b) for b in self._validator(image, hints) or []]
            found = _count_found(answer.masks, boxes)
            solved = found == len(answer.masks)
            attempts.append({"boxes": boxes, "found": found, "solved": solved})
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
        return problem_id, image, hints

    def solve(self, problem_id, point):
        """point(합성 이미지 기준 (x, y))로 숨긴 물체 하나를 짚는다. 짚는 순서는 상관없다.

        - 아직 못 찾은 물체를 맞히면 correct=True. 모두 찾으면 문제가 끝난다.
        - 이미 찾은 물체를 다시 짚으면 무시한다 (correct=False, 문제는 계속).
        - 그 밖은 오답으로 문제가 끝난다.
        끝난 문제는 폐기하고 전체 정답(answer_boxes, answer_masks)을 공개한다. box는 이번에 찾은 물체의 박스.
        """
        answer = self._answers.get(problem_id)
        if answer is None:
            return {"correct": False, "done": True, "found": 0, "box": None,
                    "answer_boxes": None, "answer_masks": None}
        i = _match(answer.masks, point, answer.found)
        if i is not None:
            answer.found.add(i)
            done = len(answer.found) == len(answer.masks)
        else:
            done = _match(answer.masks, point) is None
        if done:
            del self._answers[problem_id]
        return {
            "correct": i is not None,
            "done": done,
            "found": len(answer.found),
            "box": None if i is None else answer.boxes[i],
            "answer_boxes": answer.boxes if done else None,
            "answer_masks": answer.masks if done else None,
        }

    @property
    def history(self):
        """validator 검수 기록. validator가 없으면 비어 있다."""
        return list(self._history)

    def auto_solve_rate(self):
        """발급한 문제 중 validator가 첫 시도에 (물체를 모두 찾아) 푼 비율. 기록이 없으면 None."""
        if not self._history:
            return None
        return sum(h["attempts"][0]["solved"] for h in self._history) / len(self._history)

    def _build(self, rng):
        bg = read_image(self._backgrounds[rng.integers(len(self._backgrounds))])
        hints = [read_object(self._objects[i]) for i in rng.choice(len(self._objects), NUM_OBJECTS, replace=False)]
        image = fit_background(bg, OUTPUT_SIZE)

        occupied = np.zeros(image.shape[:2], bool)
        masks = []
        for obj in hints:
            long_side = rng.uniform(*OBJECT_SCALE_RANGE) * min(OUTPUT_SIZE)
            placed = transform_object(obj, rng, long_side, OBJECT_ANGLE_RANGE)
            composite, alpha = place_object(image, placed, rng, PLACE_MARGIN, PLACE_CANDIDATES, PLACE_TOP_K, occupied)
            image = blend_object(composite, image, alpha)
            mask = np.where(alpha > 0.5, 255, 0).astype(np.uint8)
            occupied |= mask > 0
            masks.append(mask)

        boxes = []
        for mask in masks:
            x, y, w, h = cv2.boundingRect(mask)
            boxes.append((x, y, x + w, y + h))
        return image, hints, _Answer(masks, boxes)
