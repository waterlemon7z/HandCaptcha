# HandCAPTCHA – 손 인식 모듈

웹캠에서 검지 끝을 추적해서 **문제 캔버스 좌표**로 바꾸고,
EMA 스무딩 → dwell(머무르기) 선택 → 궤적 기록까지 처리하는 모듈입니다.

| 파일 | 내용 |
|---|---|
| `hand_tracker.py` | 모듈 본체 (`HandTracker`, `HandConfig`, `HandState`) |
| `mock_captcha.py` | 연결 테스트용 `MockCaptcha` (실제 `Captcha`와 같은 `create`/`solve` 형태) |
| `demo_hand.py` | OpenCV 창 데모 (웹캠 또는 마우스) |
| `download_model.py` | MediaPipe 손 모델 파일 다운로드 |
| `tests/` | 웹캠 없이 돌아가는 단위 테스트 (pytest) |

다른 모듈과 의존 관계가 없어서 위 파일을 팀 저장소에 그대로 옮기면 됩니다.

## 설치

Python **3.12**가 기준입니다. mediapipe 1.1.1과 OpenCV 5.0으로 확인했습니다.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_model.py        # models/hand_landmarker.task (약 7.5MB)
```

> mediapipe가 `opencv-contrib-python`을 함께 설치합니다. `opencv-python`과 버전이 다르면
> `cv2`가 꼬일 수 있어서 `requirements.txt`에서 두 패키지를 같은 버전으로 고정했습니다.

> mediapipe 1.x에는 예전 `mp.solutions.hands`가 **없습니다**.
> 이 모듈은 Tasks API(`mediapipe.tasks.python.vision.HandLandmarker`, VIDEO 모드)를 쓰고,
> 그래서 `.task` 모델 파일이 필요합니다.

## 실행

```bash
python demo_hand.py              # 웹캠
python demo_hand.py --mouse      # 웹캠/모델 없이 마우스로 테스트
python demo_hand.py --smoothing 0.3 --dwell-radius 40 --dwell-time 1.2
python -m pytest                 # 단위 테스트
```

데모 키: `n` 다음 문제, `r` 다시 하기, `q`/`ESC` 종료.
선택하면 3초 뒤 자동으로 다음 문제로 넘어갑니다.

## 좌표 기준

```
캔버스 (2W + gap) × H
┌──────────┬───┬──────────┐
│  왼쪽    │gap│  오른쪽  │   W × H = 문제 이미지 원본 크기
│ (0,0)~   │   │ (W+gap,0)~│
└──────────┴───┴──────────┘
```

- **캔버스 좌표**: 좌우 이미지를 원본 크기 그대로 가로로 붙인 화면 기준 픽셀입니다.
  `HandState.point`와 궤적이 이 좌표를 씁니다.
  GUI에서 캔버스를 확대/축소해 표시한다면 그리기 직전에 배율만 곱하면 됩니다.
- **원본 이미지 좌표**: `HandState.answer`. 짚은 쪽 이미지 기준 `(x, y)` 정수이고,
  `HandState.side`(`"left"`/`"right"`)가 어느 쪽인지 알려 줍니다. 오른쪽이면 x에서 `W + gap`을 뺀 값입니다.
  gap 위를 짚으면 가까운 쪽 이미지 가장자리로 붙입니다.
- **카메라 → 캔버스**: 좌우 반전(거울)한 뒤, 카메라 중앙 활성 영역(기본 85%)을 캔버스 전체에 늘립니다.
  손끝이 카메라 가장자리까지 가지 않아도 캔버스 끝에 닿게 하려는 것입니다.

## 인터페이스

```python
from hand_tracker import HandTracker

tracker = HandTracker((W, H), gap=20, smoothing=0.4, dwell_radius=30, dwell_time=1.0)

while True:
    ok, frame = cap.read()            # 웹캠 원본 BGR 프레임 (반전하지 말 것)
    state = tracker.update(frame)     # 매 프레임

    # GUI: 커서와 게이지
    #   state.point     -> (x, y) 캔버스 좌표 또는 None
    #   state.progress  -> 0.0 ~ 1.0
    #   state.status    -> "NO_HAND" / "TRACKING" / "DWELLING" / "SELECTED"

    if state.selected:
        ok = captcha.solve(img_hash, state.answer)      # answer = (x, y) 원본 이미지 기준
        trajectory = tracker.get_trajectory()           # [(x, y, t), ...] → 진위 검출 모듈
        ...
        tracker.reset()                                 # 새 문제 시작 시

tracker.close()   # 또는 with HandTracker(...) as tracker:
```

### `HandState`

| 필드 | 타입 | 설명 |
|---|---|---|
| `point` | `(float, float)` 또는 `None` | 스무딩된 손끝 캔버스 좌표 |
| `progress` | `float` | dwell 진행률 0~1 |
| `selected` | `bool` | 선택 확정 여부. 확정되면 `reset()` 전까지 상태가 고정됩니다 |
| `answer` | `(int, int)` 또는 `None` | 확정 시 원본 이미지 기준 좌표 (dwell 동안 머문 점들의 평균) |
| `side` | `"left"` / `"right"` / `None` | 확정 시 짚은 쪽 |
| `hand_visible` | `bool` | 손 검출 여부 |
| `status` | `str` | 위 4가지 상태 상수 (`hand_tracker.STATUS_*`) |

### 궤적 `get_trajectory()`

- `[(x, y, t), ...]`. x, y는 **스무딩된** 캔버스 좌표이고, t는 `reset()` 뒤 첫 프레임부터 센 초입니다(`time.monotonic` 기준).
- 손이 보이는 프레임만 기록합니다. 손이 사라진 구간은 t 간격이 벌어지는 것으로 드러납니다.
- 선택이 확정된 뒤에는 더 쌓지 않습니다. 확정 시점의 리스트가 그 문제의 궤적 전체입니다.
- 원본 좌표가 필요하면(떨림 분석 등) 진위 검출 담당과 상의해서 `smoothing=1.0`으로 따로 기록하는 옵션을 추가할 수 있습니다.

### 웹캠 없이 쓰기

`update_point(norm_xy, t)`에 정규화 좌표(0~1, 반전이 이미 반영된 값)와 시각을 직접 넣으면
MediaPipe 없이 같은 로직을 돌릴 수 있습니다. 테스트와 데모의 `--mouse` 모드가 이 방식입니다.
`detector=`로 직접 만든 검출기(`detect(frame, timestamp_ms) -> (x, y) | None`)를 넣을 수도 있습니다.

## 설정값 (`HandConfig`)

`HandTracker(..., config=HandConfig(...))`로 넘기거나 키워드 인자로 일부만 바꿉니다.

| 이름 | 기본값 | 설명 |
|---|---|---|
| `smoothing` | 0.4 | EMA 계수 α. `s = α·p + (1-α)·s`. 작을수록 부드럽지만 반응이 느려짐. 1.0이면 스무딩 없음 |
| `dwell_radius` | 30 | 이 반경(캔버스 px) 안에 머물러야 함. 처음 들어온 점이 기준점이고, 반경을 벗어나면 그 점에서 다시 시작 |
| `dwell_time` | 1.0 | 선택 확정까지 머무는 시간(초) |
| `active_margin` | 0.075 | 카메라 각 변에서 버리는 비율. 0.075면 중앙 85% 영역 사용 |
| `lost_grace` | 0.15 | 손을 이 시간(초)보다 오래 놓쳐야 dwell을 초기화. 검출이 잠깐 끊기는 것 대비 |
| `mirror` | True | 좌우 반전 (사용자가 거울처럼 느끼게) |
| `model_path` | `models/hand_landmarker.task` | 상대 경로면 `hand_tracker.py` 위치 기준 |
| `min_detection_confidence` | 0.5 | MediaPipe 손 검출 신뢰도 |
| `min_presence_confidence` | 0.5 | MediaPipe 손 존재 신뢰도 |
| `min_tracking_confidence` | 0.5 | MediaPipe 추적 신뢰도 |

## 손이 사라졌을 때

`lost_grace`보다 오래 손이 안 보이면 스무딩과 dwell을 초기화하고
`point=None`, `progress=0`, `status="NO_HAND"`를 돌려줍니다. 궤적은 지우지 않습니다.
