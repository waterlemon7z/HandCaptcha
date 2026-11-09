# 문제 생성 모듈 (ProblemGenerator)

배경 사진에 물체 3개를 숨긴 **숨은그림찾기 문제(1280×720)** 를 만들고, 그중 1개를 정답으로 내부에 보관한다.
사용자(또는 봇)가 낸 박스 `(x1, y1, x2, y2)`가 정답 물체와 충분히 겹치는지 판정한다.

## 문제 생성 과정

1. 배경 1장과 서로 다른 물체 3개를 rng로 고른다.
2. 배경 가운데를 16:9로 잘라 1280×720으로 맞춘다. 가장 흔한 색을 기준으로 바닥 영역을 구하고, 색은 같고 어둡기만 한 곳은 그림자로 보고 바닥에 넣는다.
3. 물체마다 다음을 반복한다.
    1. 긴 변을 약 100~144px로 줄이고 ±45° 안에서 회전한다 (`warpAffine`).
    2. 후보 위치 100곳 중 물체가 사진 속 물건 밑에 20~45% 깔리고, 잘리는 선의 70% 이상이 실제 윤곽선 위에 있는 곳만 남긴다.
    3. 남은 곳에 점수(`3 × 물체 자리의 윤곽선 밀도 − 주변 고리의 윤곽선 밀도`, 낮을수록 좋음)를 매겨 상위 3곳 중 하나를 고른다. 맞는 자리가 없으면 가리지 않고 놓는다.
    4. 바닥 위 부분만 그리고, 라플라시안 피라미드로 배경과 블렌딩한다.
4. 보이는 부분 박스의 짧은 변이 48px 이상인 물체 중 하나를 정답으로 고른다. 없으면 가장 큰 물체를 고른다. 나머지 2개는 방해용이다.
5. validator가 있으면 검수한다. 정답을 찾아내면 같은 rng로 1번부터 다시 만들고, `max_attempts`번 모두 풀리면 마지막 문제를 발급한다.

## 파일

| 파일 | 내용 |
|---|---|
| `generator.py` | `ProblemGenerator`: 전체 흐름, 정답 보관, 판정, 설정 상수 |
| `compose.py` | 배경 맞추기, 물체 크기·회전, 바닥 영역 판별, 숨길 자리 고르기 |
| `blending.py` | 라플라시안 피라미드 블렌딩 (`pyrDown`/`pyrUp` 직접 구현, 홀수 크기 대응) |
| `assets.py` | 에셋 목록 읽기, 이미지 로드 (한글 경로 대응) |
| `test/` | 단위 테스트와 결과 확인용 `preview.py` |

## 사용 방법

```python
from cv.problem_gen import ProblemGenerator
from cv.verification.difficulty_check import DifficultyCheck

gen = ProblemGenerator("assets", validator=DifficultyCheck(), max_attempts=5)

problem_id, image, target = gen.create(verification_method="orb/sift")   # 정답은 반환하지 않음
# image → 화면에 표시, target → "이것을 찾으세요"로 표시

result = gen.solve(problem_id, (x1, y1, x2, y2))   # 손 인식이 준 박스로 판정
if result["correct"]:
    ...
x1, y1, x2, y2 = result["answer_box"]              # 맞든 틀리든 정답 위치를 받을 수 있음
```

### 생성자 `ProblemGenerator(asset_dir="assets", validator=None, max_attempts=5)`

| 인자 | 형식 | 기본값 | 설명 |
|---|---|---|---|
| `asset_dir` | `str` 또는 `Path` | `"assets"` | 에셋 폴더. 상대 경로면 실행 위치 기준이라 레포 루트에서 실행한다 |
| `validator` | 객체 또는 함수, 또는 `None` | `None` | 자동 풀이 검수기. `None`이면 검수하지 않고 바로 발급 |
| `max_attempts` | `int` | 5 | 검수기가 정답을 찾을 때 다시 만드는 최대 횟수 |

`objects/`의 물체가 3개보다 적으면 `ValueError`를 낸다.

### `create(seed=None, *, verification_method="orb/sift")`

| 인자 | 형식 | 기본값 | 설명 |
|---|---|---|---|
| `seed` | `int` 또는 `None` | `None` | 같은 seed면 같은 문제. `None`이면 무작위 |
| `verification_method` | `"orb/sift"` / `"orb"` / `"sift"` | `"orb/sift"` | 키워드로만 받는다. validator가 `difficult_check` 메서드를 가진 객체일 때 그대로 넘긴다. 그 밖의 값은 `ValueError` |

반환: `(problem_id, image, target)`

| 값 | 형식 | 설명 |
|---|---|---|
| `problem_id` | `str` (32자리 hex) | 문제 ID. seed와 상관없이 발급할 때마다 새로 만든다 |
| `image` | `np.ndarray` (720, 1280, 3) uint8 **BGR** | 물체 3개를 숨긴 합성 이미지. 정답 표시 없음 |
| `target` | `np.ndarray` (H, W, 4) uint8 **BGRA**, 원본 크기 | 찾아야 할 정답 물체의 원본 이미지 (투명 배경). 화면에 BGR로 띄우려면 바꿔서 쓴다 |

같은 seed는 에셋, 라이브러리 버전, 설정 상수, validator가 모두 같을 때만 같은 문제를 낸다.

### `solve(problem_id, box)`

| 인자 | 형식 | 설명 |
|---|---|---|
| `problem_id` | `str` | `create`가 준 ID |
| `box` | `(x1, y1, x2, y2)` int 또는 float | 합성 이미지 기준 박스. 좌표 순서가 뒤집혀도 된다 |

반환: `dict`

| 키 | 형식 | 설명 |
|---|---|---|
| `"correct"` | `bool` | 정답 여부 |
| `"answer_box"` | `(x1, y1, x2, y2)` int 또는 `None` | 정답 물체의 **보이는 부분** 박스 |
| `"answer_mask"` | `np.ndarray` (720, 1280) uint8 (0/255) 또는 `None` | 정답 물체의 보이는 부분 마스크. 결과 화면 하이라이트용 |

- **한 번만 판정:** 판정한 ID는 바로 폐기된다. 맞든 틀리든 정답을 함께 돌려준다.
- **실패 처리:** 없는 ID나 이미 판정한 ID면 `{"correct": False, "answer_box": None, "answer_mask": None}`
- **판정 기준:** 입력 박스와 정답 박스의 IoU가 0.3 이상이면 정답이다. 원래 정답 박스와 사방으로 16px 넓힌 박스 중 IoU가 더 큰 쪽을 쓴다.

### validator 형식

| 형태 | 호출 | 반환 |
|---|---|---|
| `difficult_check` 메서드가 있는 객체 (`DifficultyCheck`) | `validator.difficult_check(image, target, method=verification_method)` | `[(x1, y1, x2, y2), ...]`. 못 찾으면 `[]` |
| 함수 | `validator(image, target)` | `[(x1, y1, x2, y2), ...]`, `[]` 또는 `None` |

- **입력:** `image`는 BGR 합성 이미지, `target`은 정답 물체 1개의 BGRA 원본이다. 방해용 물체는 넘기지 않는다. `target[..., 3]`은 특징점 검출 마스크로 쓸 수 있다.
- **판정:** 반환한 박스 중 하나라도 위 판정 기준을 넘으면 "봇이 풀었다"로 보고 문제를 다시 만든다. 박스가 어떤 물체인지는 몰라도 된다.

### 검수 기록

- **`gen.history`:** 문제마다 기록 1개의 목록이다. validator가 없으면 비어 있다.
  ```python
  {"problem_id": "3f2a...", "seed": 1201,
   "attempts": [{"boxes": [(412, 96, 470, 150)], "solved": True},
                {"boxes": [], "solved": False}],
   "cap_reached": False}      # max_attempts번 모두 풀렸으면 True
  ```
- **`gen.auto_solve_rate()`:** 첫 시도에 validator가 정답을 찾은 비율(0~1)이다. 기록이 없으면 `None`이다.

### 좌표 규칙

- 모든 좌표는 **합성 이미지 기준**이다(x 0~1279, y 0~719, 왼쪽 위가 원점).
- 박스의 `x2`, `y2`는 박스 바로 바깥(슬라이스 끝)이다. `image[y1:y2, x1:x2]`가 박스 안이다.

### 설정값 (`generator.py` 상수)

| 상수 | 값 | 설명 |
|---|---|---|
| `OUTPUT_SIZE` | (1280, 720) | 출력 크기 (w, h) |
| `NUM_OBJECTS` | 3 | 합성하는 물체 수 (정답 1 + 방해용 2) |
| `OBJECT_SCALE_RANGE` | 0.14 ~ 0.20 | 물체 긴 변(회전 전) ÷ 720 (약 100~144px) |
| `OBJECT_ANGLE_RANGE` | −45 ~ 45° | 물체 회전 각도 |
| `PLACE_MARGIN` | 0.05 | 가장자리 여백 비율 |
| `PLACE_CANDIDATES`, `PLACE_TOP_K` | 100, 3 | 위치 후보 수, 점수 상위 몇 곳 중에서 고를지 |
| `OCCLUDE_RANGE` | 0.2 ~ 0.45 | 물체가 사진 속 물건 밑에 가려지는 비율 |
| `BLEND_LEVELS`, `BLEND_FEATHER_SIGMA` | 4, 2.0 | 피라미드 단계 상한, 블렌딩 마스크 흐림(px) |
| `HIT_IOU`, `HIT_PAD_PX` | 0.3, 16 | 판정 IoU 기준, 정답 박스를 넓히는 픽셀 수 |
| `MIN_ANSWER_SIDE_PX` | 48 | 정답으로 고를 물체의 보이는 박스 짧은 변 최소값 |

바닥·그림자·윤곽선 기준(`FLOOR_COLOR_DIST`, `FLOOR_SHADOW`, `CUT_EDGE_RATIO`, `CLEARANCE_PX`, `SURROUND_PX`)은 `compose.py` 위쪽에 있다. 상수를 바꾸면 같은 seed라도 다른 문제가 나온다.

## 에셋

`asset_dir` 아래에 두 폴더가 있어야 한다. 파일은 이름순으로 정렬해서 쓴다.

| 폴더 | 형식 | 개수 |
|---|---|---|
| `backgrounds/` | `.jpg` / `.jpeg` / `.png`, 크기·비율 자유 | 10장 |
| `objects/` | **투명 배경 PNG(BGRA)**만. 4채널이 아니면 `ValueError` | 42개 |

사진은 모두 [Unsplash](https://unsplash.com)에서 가져왔고 [Unsplash License](https://unsplash.com/license)를 따른다. 파일별 사진작가와 원본 링크는 [`assets/SOURCES.md`](../../assets/SOURCES.md)에 있다.

| 종류 | 가공 | 사진 |
|---|---|---|
| 배경 10장 | 가로 1600px로 축소. 코드에서 1280×720으로 변환 | 책상 4, 공예 재료 2, 장난감 3, 책장 1 |
| 물체 기본 20개 | 흰 배경 사진을 [rembg](https://github.com/danielgatis/rembg)(`isnet-general-use`)로 배경 제거 → 물체 영역 크롭 → 긴 변 500px 이하 투명 PNG | 집게, 동전 2, 주사위 2, 이어폰, 지우개, 풀, 형광펜, 열쇠, 립스틱, 클립, 연필, 고무 오리, 숟가락, 선글라스, 실, 장난감 자동차 2, 렌치 |
| 물체 추가 22개 | 위와 같음 | 사과, 공, 양초, 시계, 커피, 크레용, 큐브, 공룡, 마우스, 붓, 솔방울, 가위, 조개껍데기 2, 모양 맞추기 장난감, 스테이플러, 다육식물, 줄자, 호두, 마스킹 테이프, 손목시계, 털실 |

파일 이름은 `bg_<장면>_<번호>.jpg`, `obj_<물체>_<번호>.png` 규칙을 따른다.

## 실행과 테스트

레포 루트에서 실행한다.

```bash
.venv\Scripts\python.exe -m pytest cv/problem_gen/test -q
```

```bash
.venv\Scripts\python.exe -m cv.problem_gen.test.preview --out preview_out --seeds 0 1 2
```

`preview_out/`에 `problem_000.png`(합성 이미지, 정답 표시, 찾을 물체)와 `blend_000.png`(블렌딩 방식별 확대 비교)가 저장된다.

## 설계 결정과 한계

**결정**
- **물체를 사진 속 물건 밑에 가려 합성:** 전체를 붙이면 눈에 잘 띄고, 가려진 만큼 ORB/SIFT가 쓸 특징점도 줄어든다.
- **잘리는 선이 실제 윤곽선 위에 있을 때만 가림:** 그림자 경계나 바닥 얼룩에서 잘리면 가려진 게 아니라 잘린 것처럼 보인다.
- **정답 박스·마스크는 보이는 부분 기준:** 사람은 보이는 부분에 박스를 그린다.
- **IoU 판정 + 16px 여유:** 손 박스가 어긋나거나 크고 작게 잡혀도 인정한다. "정답 박스를 덮은 비율"로 판정하면 화면 전체를 덮는 박스도 정답이 된다. 100px 정답에 무작위 박스 2만 개(크기 50~600px)를 넣어 보니 우연히 맞을 확률은 IoU 방식 0.4%, 덮은 비율 방식 25.6%였다.
- **가늘거나 많이 가려진 물체는 정답에서 제외:** 정답 박스가 작으면 손 박스와 IoU가 너무 낮게 나온다.

**한계**
- 흰 책상 위 흰 키보드처럼 바닥과 색이 같은 물건은 가리개로 쓰이지 않는다.
- 책장처럼 바닥이 없는 배경에서는 합성 결과가 어색하다.
- 그림자 위에 놓인 물체를 어둡게 처리하지 않는다.
- 정답과 바로 옆 방해 물체를 한꺼번에 덮는 박스도 정답으로 판정될 수 있다.
