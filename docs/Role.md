## 클래스 별 행동
### 문제생성 
- Input: (x1,y1,x2,y2) 해당 좌표의 정답여부 (boolean)
- Output: Image 1280 x 720, np.array (BGR)

### 난이도검수
- Input: np.array (BGR) 합성된 큰 이미지, 합성에 사용한 작은 이미지 
- Output: 검출된 영역의 좌표들 (x1,y1,x2,y2)

### 손인식
- Input: frame 
- Output: 현재 손 좌표, 손 시리얼 데이터 (x, y), np.array(2D)

### 진위검출
- Input: 손인식 `HandResult` (state == "done"일 때)
  - `raw_trajectory` np.array (N, 2) float, EMA 보정 전 손 좌표 (문제 이미지 픽셀)
  - `timestamps` np.array (N,), 기록 시작 = 0초
  - `frames` 검지 끝 주변 crop 이미지 목록, `frame_size` 카메라 프레임 (w, h)
- Output: (is_human: Boolean, reason: str)
  - reason은 봇 판정 이유 (디버깅·발표용, 화면에는 띄우지 않기)
- 호출: `check(result)` — `cv/bot_detection/detector.py`

### CV 입력
- Input: 카메라
- Output: Frame np.array()