# 손인식 모듈 (HandTracker)

카메라 프레임에서 손을 찾아 **문제 이미지(1280×720) 좌표**로 바꾸고,
**주먹을 쥐면 기록 시작 → 손을 움직임 → 1초 멈추면 완료**하여
**움직인 경로**(`trajectory`)와 **마지막에 멈춘 위치**(`stop_position`)를 따로 돌려준다.

| 파일 | 내용 |
|---|---|
| `hand_tracker.py` | `HandTracker` 클래스와 결과 `HandResult` |
| `demo.py` | 웹캠 데모 |
| `download_model.py` | MediaPipe 손 모델 다운로드 |
| `tests/` | 카메라 없이 돌아가는 테스트 |
| `docs/decisions.md` | 정한 기준과 이유 |

## 설치

Python 3.12 기준이다. mediapipe 1.1.1은 Python 3.9~3.14를 지원한다.

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
```

> mediapipe 1.x에는 예전 `mp.solutions.hands`가 없어서 Tasks API(`HandLandmarker`)와
> 모델 파일 `models/hand_landmarker.task`(약 7.8MB)를 쓴다.

## 데모 실행

```bash
.venv\Scripts\python.exe demo.py
```

- 손을 펴서 보이면 문제 이미지 위에 커서가 나온다 (회색 = idle).
- **주먹**을 쥐면 기록이 시작된다 (주황색). 기록 중에는 손을 펴도 된다.
- 찾을 대상으로 손을 옮겨 **1초 멈추면** 완료된다. 멈춘 위치에 초록 원과 십자 표시가 나온다.
- 이동 경로는 화면에 그리지 않는다. 완료되면 멈춘 위치, 경로 점 개수, 손끝 프레임 장수를 터미널에 출력한다.
  파일로는 저장하지 않는다. 값은 모두 `result`로 넘어간다.
- 완료되면 그 상태로 멈춰 있다. 화면 아래 **다시하기** 버튼을 클릭하거나 `r`을 누르면 처음부터 다시 한다.
- `q` 종료
- `samples/problem.png`, `samples/hint.png`가 있으면 그 이미지를 쓰고, 없으면 임시 이미지를 만든다.

## 테스트

```bash
.venv\Scripts\python.exe -m pytest
```

## 인터페이스

### 입력
| 이름 | 형식 | 설명 |
|---|---|---|
| `frame` | `np.ndarray` BGR | 카메라 프레임 (CV 입력 클래스가 줌). 손 검출은 이것으로만 한다 |
| `problem_img` | `np.ndarray` BGR, 1280×720 | 사용자가 손으로 정답 영역을 지정하는 이미지 |
| `hint_img` | `np.ndarray` BGR, 크기 자유 | 찾을 대상을 보여 주는 작은 이미지. 표시만 하고 판정에는 안 씀 |

원본 배열은 수정하지 않는다.

### 출력 `HandResult` (dataclass)
| 필드 | 형식 | 설명 |
|---|---|---|
| `position` | `(x, y)` 또는 `None` | **검지 끝**(랜드마크 8) 좌표, EMA 보정본. 손이 없거나 문제 이미지 밖이면 `None` |
| `raw_position` | `(x, y)` 또는 `None` | 같은 프레임의 **EMA 적용 전 원본** 좌표. `position`이 `None`이면 `None` |
| `hand_status` | `"none"` / `"open"` / `"fist"` | 손 모양 |
| `state` | `"idle"` / `"recording"` / `"done"` | 기록 상태 |
| `trajectory` | `np.ndarray (N, 2)` int32 | 기록 시작(주먹)부터 지금(또는 완료)까지의 궤적 전체(EMA 보정본), 각 행이 `(x, y)`. 길이 제한 없음 |
| `raw_trajectory` | `np.ndarray (N, 2)` int32 | `trajectory`와 같은 프레임들의 **원본** 좌표. `trajectory[i]`와 `raw_trajectory[i]`가 짝 |
| `timestamps` | `np.ndarray (N,)` | 각 점의 시각(초), 기록 시작 = 0. 진위검출용 |
| `stop_position` | `(x, y)` 또는 `None` | `state == "done"`일 때만. **마지막에 멈춘 위치** = 멈춤 판정에 쓴 최근 1초 동안 점들의 평균 |
| `box` | `(x1, y1, x2, y2)` 또는 `None` | `state == "done"`일 때만. **경로 전체**를 감싸는 사각형. 항상 `x1 < x2`, `y1 < y2`, 이미지 안. 화면에는 그리지 않음 |
| `frames` | `list[np.ndarray]` 또는 `None` | `state == "done"`일 때만. 기록 중 모은 **검지 끝 주변 crop**(기본 128×128) 또는 1/4 축소본. `process(frame)`으로 돌리면 `trajectory`와 같은 개수 |
| `frame_timestamps` | `np.ndarray` 또는 `None` | `state == "done"`일 때만. `frames` 각각의 시각(초), `timestamps`와 같은 기준 |

**좌표는 모두 문제 이미지 기준**(x 0~1279, y 0~719)이다. 원본 좌표도 이미지 범위 안으로 자른다.
`frames`는 카메라 프레임에서 잘라 낸 이미지이고, 화면과 같은 방향이 되도록 좌우 반전돼 있다.

### 상태 전환
```
idle ──(문제 이미지 안에서 주먹)──▶ recording ──(최근 1초 이동 범위 < 15px)──▶ done
  ▲                                     │                                      │
  └──(손이 0.5초 이상 안 보이거나 문제 이미지 밖)┘                             │
  └─────────────────────────────── reset() ◀──────────────────────────────────┘
```

### 화면 `render(result)`
`(720, 1280 + 320, 3)` BGR 이미지를 돌려준다.
- 왼쪽 1280×720: 문제 이미지 + 손 커서, 상태 글자, 완료 시 멈춘 위치 표시(초록 원·십자).
  궤적 선과 경로 box는 그리지 않는다.
- 오른쪽 320px: 힌트 패널 ("찾을 대상" 글자 + 비율 유지해 가운데 배치한 힌트 이미지).
  **어떤 오버레이도 그리지 않으며 항상 원본 그대로다.** 손이 힌트 패널 쪽으로 가면 무시한다.

## 통합 클래스에서 쓰는 예

```python
from hand_tracker import HandTracker

tracker = HandTracker(problem_img, hint_img)     # 문제 하나당 하나 (또는 set_images로 교체)

while True:
    frame = camera.read()                         # CV 입력 클래스가 주는 BGR 프레임
    result = tracker.process(frame)
    cv2.imshow("HandCAPTCHA", tracker.render(result))   # 또는 GUI에 맞게 표시

    if result.state == "done":
        stop = result.stop_position               # (x, y) 마지막에 멈춘 위치 → 정답 판정
        path_box = result.box                     # (x1, y1, x2, y2) 경로 전체 범위 (필요하면)
        trajectory = result.trajectory            # (N, 2) 움직인 경로 (보정본) → 진위검출
        raw = result.raw_trajectory               # (N, 2) 같은 경로의 원본 (미세 떨림 분석용)
        times = result.timestamps                 # (N,)  → 진위검출
        frames = result.frames                    # 손끝 crop 이미지 목록 → 진위검출 (필요하면)
        frame_times = result.frame_timestamps     # frames 각각의 시각
        break

# 다음 문제
tracker.set_images(next_problem_img, next_hint_img)   # 상태도 초기화됨
# 같은 문제를 다시 풀게 할 때는 tracker.reset()
tracker.close()                                        # 프로그램 끝날 때
```

카메라 없이 로직만 쓰려면 `tracker.update((x, y), "fist", t)`에 문제 이미지 좌표를 직접 넣거나,
`tracker.process_landmarks(landmarks, t, frame)`에 랜드마크 21개 `[(x, y), ...]`(정규화 좌표)를 넣으면 된다.
`frame`을 빼면 프레임 버퍼(`frames`)는 비어 있다.

## 설정값 (생성자 키워드 인자)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `stop_seconds` | 1.0 | 멈춤 판정 시간(초) |
| `stop_pixels` | 15 | 이 시간 동안 이동 범위가 이 값 **미만**이면 완료 |
| `lost_seconds` | 0.5 | 기록 중 손이 이 시간 이상 안 보이면 취소 |
| `alpha` | 0.5 | 좌표 떨림 보정(EMA) 계수. 1.0이면 보정 없음 |
| `min_folded` | 3 | 주먹 판정: 엄지 제외 접힌 손가락 수 |
| `anchor` | `"index_tip"` | 커서 기준점. `"index_tip"` = 검지 끝(랜드마크 8), `"palm"` = 손바닥 중심 |
| `clip_mode` | `"crop"` | 기록 중 모을 프레임. `"crop"` = 검지 끝 주변, `"quarter"` = 카메라 프레임 1/4 축소본, `None` = 모으지 않음 |
| `crop_size` | 128 | `"crop"`일 때 잘라 낼 크기 (카메라 프레임 픽셀) |
| `mirror` | True | 좌우 반전 (셀카처럼) |
| `panel_width` | 320 | 힌트 패널 폭 |
| `model_path` | `models/hand_landmarker.task` | MediaPipe 모델 경로 |

판정 기준
- 위치: 검지 끝 = 랜드마크 8 (`anchor="palm"`이면 랜드마크 0, 5, 9, 13, 17의 평균)
- 주먹: 엄지를 뺀 네 손가락 중 `min_folded`개 이상이 접힘 (손가락 끝이 PIP 관절보다 손목에 가까우면 접힘)

자세한 기준과 이유는 [docs/decisions.md](docs/decisions.md)에 있다.
