"""팀원 구현 전 연결 확인용 mock. 실제 인식/인증 알고리즘이 아니다."""

import numpy as np


class MockHandRecognizer:
    def recognize(self, frame):
        # 실제 구현은 ((x, y), np.ndarray(2D))를 반환한다.
        # 기본 mock은 손 미검출로 두어 임의의 인식 이벤트를 만들지 않는다.
        return None, np.empty((0, 2), dtype=float)


class MockProblemGenerator:
    def requestCaptcha(self, region):
        self.region = region
        image = np.full((720, 1280, 3), 240, dtype=np.uint8)
        x1, y1, x2, y2 = region
        image[y1:y2, x1:x2] = (220, 150, 70)
        self.small_image = image[y1:y2, x1:x2].copy()
        return image

    def isAnswer(self, region):
        x1, y1, x2, y2 = self.region
        x, y = region[:2]
        return x1 <= x <= x2 and y1 <= y <= y2


class MockDifficultyChecker:
    def inspect(self, image, small_image):
        # mock 생성기의 고정 영역. 실제 구현은 검출한 영역 목록을 반환한다.
        return [(480, 240, 800, 480)]


class MockBotDetector:
    def is_bot(self, serial):
        # 봇 여부 Boolean 인터페이스만 제공한다. 실제 인증에 사용하지 않는다.
        return False
