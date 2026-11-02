# 카메라 예제

`Camera`는 별도 Qt 스레드에서 OpenCV 카메라를 읽습니다.

- `start()`: 촬영 시작 (QThread에서 제공)
- `stop()`: 촬영 중지 및 카메라 해제 대기
- `frame_ready`: BGR 형식 `numpy.ndarray` 전달
- `error_occurred`: 장치 열기 또는 프레임 읽기 오류 메시지 전달
- `finished`: 스레드 종료 알림 (QThread에서 제공)

`main.py`에서 생성한 뒤 현재 controller의 `camera` 인자로 전달하세요.

```python
from cv.camera import Camera

camera = Camera(device_index=0, width=640, height=480)
controller = Controller(window, camera)
app.aboutToQuit.connect(camera.stop)

# 연결을 모두 마친 다음 촬영 시작
window.show()
camera.start()
sys.exit(app.exec())
```

controller에서는 시그널을 메서드에 연결합니다.

```python
# Controller.__init__ 안에서
self.camera.frame_ready.connect(self.on_frame)
self.camera.error_occurred.connect(self.on_camera_error)

# Controller의 메서드
@Slot(object)
def on_frame(self, frame):
    self.view.set_captcha(frame)

@Slot(str)
def on_camera_error(self, message):
    print(message)
```

이 예시는 기존 `bigImg` 영역에 카메라 영상을 표시합니다. 캡차 이미지도 같은
영역을 사용하므로, 두 영상을 동시에 표시하려면 별도 표시 메서드를 추가하세요.
macOS에서는 실행하는 터미널 또는 IDE에 카메라 접근 권한이 필요합니다.
요청한 해상도의 적용 여부는 장치에 따라 다릅니다.
