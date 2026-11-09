"""자동 풀이 검증기 선택 및 실행.

ProblemGenerator(validator=DifficultyCheck())로 연결한다.
입력은 uint8 BGR/BGRA/흑백 배열, 출력은 문제 좌표의 xyxy 박스 리스트.
정답 비교와 문제 재생성은 생성기가 담당한다.
"""

from cv.verification.orb_sift import OrbSiftChecker
from cv.verification.owlv2 import Owlv2Checker


class DifficultyCheck:
    def __init__(self):
        self._orb_sift = OrbSiftChecker()
        self._owlv2 = None

    def difficult_check(self, image, target, *, method="orb/sift"):
        """선택한 검출기의 후보 박스를 반환한다. 미검출은 [].

        auto는 ORB와 SIFT를 모두 실행한다. orb와 sift는 해당 방법만 실행한다.
        owlv2는 대상 이미지를 참조해 검출한다. 최초 선택 시 모델을 로드한다.
        x2/y2는 슬라이스 끝 좌표이며, 미검출은 인간 난이도 보장이 아니다.
        """
        if method in ("orb/sift", "orb", "sift"):
            return self._orb_sift.detect(image, target, method=method)
        elif method == "owlv2":
            if self._owlv2 is None:
                self._owlv2 = Owlv2Checker()
            return self._owlv2.detect(image, target)
        raise ValueError("method는 orb/sift, orb, sift, owlv2 중 하나여야 합니다.")
