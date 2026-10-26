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
- Input: np.array(2D) 손 시리얼 데이터
- Output: 봇여부 Boolean

### CV 입력
- Input: 카메라
- Output: Frame np.array()