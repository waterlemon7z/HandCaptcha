# 손인식 모듈 (HandTracker)

카메라 프레임에서 손을 찾아 검지 끝 위치를 **문제 이미지(1280×720) 좌표**로 바꾸고,
손동작으로 사용자가 고른 영역을 돌려줍니다.
정답 판정과 진위검출은 다른 모듈이 하고, 이 모듈은 그에 필요한 데이터만 넘깁니다.

## 동작

1. 손을 편 채로 정답 근처로 이동합니다. 이때는 기록하지 않습니다.
2. **주먹을 쥐었다가 검지를 펴면** 기록을 시작합니다.
3. 정답 위에서 **1초 멈추면**(1초 동안 15px 미만 이동) 완료(`state == "done"`)됩니다.
4. 기록 중 손이 0.5초 이상 안 보이거나 문제 이미지 밖으로 나가면 기록을 취소하고 처음으로 돌아갑니다.

```
idle ──(주먹 → 검지 펴기)──▶ recording ──(1초 동안 15px 미만 이동)──▶ done
  ▲                              │                                    │
  └──(손이 0.5초 이상 안 보임)─────┘                                    │
  └────────────────────── reset() / set_images() ◀─────────────────────┘
```

## 파일

| 파일 | 내용 |
|---|---|
| `hand_tracker.py` | `HandTracker` 클래스, 결과 `HandResult`, `get_stop_box()` |
| `download_model.py` | MediaPipe 손 모델 다운로드 |
| `test/demo.py` | 웹캠 데모 |
| `test/test_hand_tracker.py` | 카메라 없이 돌아가는 테스트 |

## 실행 환경

- 프로젝트 루트 `pyproject.toml`의 패키지 (mediapipe 1.1.1, opencv-python, numpy, pillow)
- MediaPipe 손 모델 파일 `models/hand_landmarker.task` (약 7.8MB, git에는 올리지 않음). 처음 한 번 받아야 합니다.

```bash
python cv/hand_tracking/download_model.py
```

> mediapipe 1.x에는 예전 `mp.solutions.hands`가 없어서 Tasks API(`HandLandmarker`)를 씁니다.

## 사용 방법

```python
from cv.hand_tracking.hand_tracker import HandTracker, get_stop_box

tracker = HandTracker()
tracker.set_images(problem_img, hint_img)   # 두 이미지가 모두 들어가야 동작 (tracker.ready == True)

result = tracker.process(frame)             # 매 프레임. 이미지 준비 전에는 None
canvas = tracker.render(result)             # 커서·상태가 그려진 문제 이미지 (원본은 수정 안 함)

if result.state == "done":
    box = get_stop_box(result)              # 1초 멈춘 영역 박스 → 정답 판정
    # result를 그대로 진위검출 check(result)에 넘김

tracker.reset()                             # 같은 문제를 다시 풀 때
tracker.set_images(next_problem, next_hint) # 다음 문제 (상태도 초기화)
tracker.close()                             # 프로그램 끝날 때
```

### 입력

| 이름 | 형식 | 설명 |
|---|---|---|
| `frame` | `np.ndarray` BGR | 카메라 프레임. 손 검출은 이것으로만 합니다 |
| `problem_img` | `np.ndarray` BGR, 1280×720 | 사용자가 손으로 정답을 짚는 이미지 |
| `hint_img` | `np.ndarray` BGR, 크기 자유 | 찾을 대상 이미지. 표시만 하고 판정에는 안 씀 |

## 반환값 `HandResult`

좌표는 모두 **문제 이미지 기준**(x 0~1279, y 0~719)입니다. `frame_size`만 카메라 프레임 기준입니다.

| 필드 | 형식 | 설명 | 쓰는 곳 |
|---|---|---|---|
| `state` | `"idle"` / `"recording"` / `"done"` | 기록 상태 | 통합(worker): `"done"`이 되면 정답 판정 시작 |
| `position` | `(x, y)` int 또는 `None` | 현재 **검지 끝** 좌표 (떨림 보정본). 손이 없거나 문제 이미지 밖이면 `None` | GUI(controller): 손 인식 여부 확인, `render()`: 커서 |
| `hand_status` | `"none"` / `"open"` / `"fist"` | 손 모양 | `render()`: 상태 글자 |
| `raw_trajectory` | `np.ndarray (N, 2)` float64 | 기록한 경로의 **떨림 보정 전 원본** 좌표. 반올림하지 않음 | **진위검출**: 멈춤 떨림·경로 분석 |
| `timestamps` | `np.ndarray (N,)` | 각 점의 시각(초), 기록 시작 = 0 | **진위검출** |
| `frames` | `list[np.ndarray]` 또는 `None` | `done`일 때만. 기록 중 모은 **검지 끝 주변 crop**(기본 128×128) | **진위검출**: 영상 검사 |
| `frame_size` | `(w, h)` 또는 `None` | 카메라 프레임 크기 (가로, 세로 순서). `reset()`해도 유지 | **진위검출**: crop 위치 계산 |
| `trajectory` | `np.ndarray (N, 2)` int32 | 기록한 경로 (떨림 보정본). `raw_trajectory[i]`와 같은 프레임 | 지금은 안 씀 (참고용, 녹화 저장) |
| `stop_position` | `(x, y)` int 또는 `None` | `done`일 때만. 1초 멈춘 구간 점들의 평균 위치 | `render()`: 완료 표시 |
| `box` | `(x1, y1, x2, y2)` 또는 `None` | `done`일 때만. 시작부터 멈출 때까지 **경로 전체**를 감싸는 박스 | 지금은 안 씀 (참고용) |
| `raw_position` | `(x, y)` float 또는 `None` | `position`의 떨림 보정 전 원본 | 지금은 안 씀 (참고용) |
| `frame_timestamps` | `np.ndarray` 또는 `None` | `done`일 때만. `frames` 각각의 시각 | 지금은 안 씀 (참고용) |

### `get_stop_box(result)`: 정답 판정용 박스

```python
box = get_stop_box(result)   # (x1, y1, x2, y2) int, done이 아니면 None
```

- 마지막 1초(멈춘 구간) 동안의 검지 끝 점들을 감싸는 박스입니다.
- 멈춘 구간은 15px도 안 되게 작아서, 가운데를 기준으로 **최소 100×100px**까지 넓힙니다. 크기는 `min_size`로 바꿀 수 있습니다.
- 문제 이미지 밖으로 나가지 않게 자릅니다.
- 문제 생성 모듈의 `solve(problem_id, box)`에 넘기면 정답 박스와의 IoU로 판정합니다.

> `box` 필드와 헷갈리지 않게 주의하세요. `box`는 시작 위치부터 멈춘 곳까지 이동한 범위 전체라서 정답 판정에는 맞지 않습니다.

### `render(result, only_captcha=True)`

- 기본(`only_captcha=True`): 문제 이미지(1280×720)에 손 커서, 상태 글자, 완료 표시를 그려 돌려줍니다.
- `only_captcha=False`: 오른쪽에 힌트 패널(320px)을 붙여 돌려줍니다. 힌트 패널에는 아무것도 그리지 않습니다.
- 이동 경로는 그리지 않습니다. 이미지 준비 전에는 `None`입니다.

## 설정 (생성자 키워드 인자)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `stop_seconds` | 1.0 | 멈춤 판정 시간(초) |
| `stop_pixels` | 15 | 이 시간 동안 이동 범위가 이 값 **미만**이면 완료 |
| `lost_seconds` | 0.5 | 기록 중 손이 이 시간 이상 안 보이면 취소 |
| `alpha` | 0.5 | 떨림 보정(EMA) 계수. 1.0이면 보정 없음 |
| `min_folded` | 3 | 주먹 판정: 엄지를 뺀 네 손가락 중 접힌 개수 |
| `start_gesture` | `"point"` | 기록 시작 동작. `"point"` = 주먹 → 검지 펴기, `"fist"` = 주먹을 쥐는 순간 |
| `anchor` | `"index_tip"` | 커서 기준점. `"index_tip"` = 검지 끝(랜드마크 8), `"palm"` = 손바닥 중심 |
| `clip_mode` | `"crop"` | 기록 중 모을 영상. `"crop"` = 검지 끝 주변, `"quarter"` = 1/4 축소본, `None` = 안 모음 |
| `crop_size` | 128 | `"crop"`일 때 잘라 낼 크기 (카메라 프레임 픽셀) |
| `panel_width` | 320 | 힌트 패널 폭 |
| `model_path` | `models/hand_landmarker.task` | MediaPipe 모델 경로 |

판정 기준
- 주먹: 엄지를 뺀 네 손가락 중 `min_folded`개 이상이 접힘 (손가락 끝이 PIP 관절보다 손목에 가까우면 접힘)
- 검지 펴짐: 검지 끝이 PIP 관절보다 손목에서 멂. 검지만 편 손은 세 손가락이 접혀 있어 `hand_status`가 `"fist"`로 나올 수 있습니다.

## 데모와 테스트

프로젝트 루트에서 실행합니다.

```bash
python -m cv.hand_tracking.test.demo
python -m pytest cv/hand_tracking/test
```

- 데모: 손을 편 채로 이동 → 주먹을 쥐었다가 검지 펴기 → 1초 멈춤. 화면 아래 **다시하기** 버튼 또는 `r`로 다시 하고, `q`로 종료합니다.
- 테스트는 카메라 없이 가짜 손 좌표로 상태 전환, 멈춤 판정, 좌표 변환, 박스 계산을 확인합니다.
