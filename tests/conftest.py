# 프로젝트 루트의 모듈(hand_tracker 등)을 테스트에서 import할 수 있게 경로 추가
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
