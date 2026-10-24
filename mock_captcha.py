"""문제 생성 모듈이 완성되기 전 연결 테스트용 MockCaptcha.

실제 Captcha와 같은 형태:
    create() -> (img_hash, left_img, right_img)
    solve(img_hash, answer) -> bool      # answer = (x, y), 원본 이미지 기준
정답 박스는 내부에만 보관하고 밖으로 넘기지 않는다.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2)


class MockCaptcha:
    def __init__(
        self,
        size: tuple[int, int] = (480, 360),
        answer_box: Box = (300, 90, 380, 170),
        seed: Optional[int] = None,
    ):
        """
        size: 이미지 크기 (W, H)
        answer_box: 고정 정답 박스. 오른쪽 이미지의 이 영역에만 다른 물체를 그린다
        seed: 배경 무늬 난수 시드
        """
        self.size = size
        self._answer_box = answer_box
        self._rng = np.random.default_rng(seed)
        self._answers: dict[str, Box] = {}

    def create(self) -> tuple[str, np.ndarray, np.ndarray]:
        w, h = self.size
        left = self._random_background(w, h)
        right = left.copy()

        # 정답 박스 안에 물체 하나를 추가해서 차이를 만든다
        x1, y1, x2, y2 = self._answer_box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max(5, min(x2 - x1, y2 - y1) // 2 - 6)
        color = tuple(int(c) for c in self._rng.integers(0, 256, 3))
        cv2.circle(right, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(right, center, radius, (0, 0, 0), 2, cv2.LINE_AA)

        img_hash = hashlib.sha1(left.tobytes() + right.tobytes()).hexdigest()[:16]
        self._answers[img_hash] = self._answer_box
        return img_hash, left, right

    def solve(self, img_hash: str, answer: tuple[int, int]) -> bool:
        """정답 여부. 같은 문제는 한 번만 풀 수 있다 (재시도 방지)."""
        box = self._answers.pop(img_hash, None)
        if box is None or answer is None:
            return False
        x, y = answer
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _random_background(self, w: int, h: int) -> np.ndarray:
        img = np.full((h, w, 3), 235, np.uint8)
        for _ in range(25):
            color = tuple(int(c) for c in self._rng.integers(40, 220, 3))
            x, y = int(self._rng.integers(0, w)), int(self._rng.integers(0, h))
            s = int(self._rng.integers(10, 45))
            if self._rng.random() < 0.5:
                cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)
            else:
                cv2.circle(img, (x, y), s, color, -1, cv2.LINE_AA)
        return img
