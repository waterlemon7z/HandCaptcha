"""MediaPipe 손 랜드마크 모델(hand_landmarker.task)을 models/ 폴더에 내려받는다."""
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def main() -> int:
    if MODEL_PATH.exists():
        print(f"이미 있습니다: {MODEL_PATH}")
        return 0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"내려받는 중: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"저장 완료: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
