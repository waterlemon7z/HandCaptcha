import cv2
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot


class Controller(QObject):
    generate_requested = Signal(int)
    verify_requested = Signal(int, object, object)

    def __init__(self, view, camera, task_worker):
        super().__init__()
        self.view = view
        self.camera = camera
        self.task_worker = task_worker
        self.latest_frame = None
        self.position = None
        self.serial = None
        self.captcha = None # 실제 captcha 이미지
        self.regions = []
        self.state = "READY"
        self._request_id = 0
        self._started = False
        self._closing = False
        self._camera_failed = False
        self.cv_thread = QThread(self)
        self.cv_thread.setObjectName("CV Worker")
        self.task_thread = QThread(self)
        self.task_thread.setObjectName("Task Worker")
        camera.moveToThread(self.cv_thread)
        task_worker.moveToThread(self.task_thread)
        self.cv_thread.started.connect(camera.start)
        # finished는 해당 작업 스레드에서 발생한다. 이벤트 루프가 끝나도 해제한다.
        self.cv_thread.finished.connect(camera.stop, Qt.ConnectionType.DirectConnection)
        self.cv_thread.finished.connect(camera.deleteLater)
        self.task_thread.finished.connect(task_worker.deleteLater)
        camera.finished.connect(self.cv_thread.quit, Qt.ConnectionType.DirectConnection)
        self.generate_requested.connect(task_worker.generate, Qt.ConnectionType.QueuedConnection)
        self.verify_requested.connect(task_worker.verify, Qt.ConnectionType.QueuedConnection)
        camera.frame_ready.connect(self.on_frame, Qt.ConnectionType.QueuedConnection)
        camera.hand_updated.connect(self.on_hand, Qt.ConnectionType.QueuedConnection)
        camera.error_occurred.connect(self.on_camera_error)
        task_worker.captcha_ready.connect(self.on_captcha)
        task_worker.verification_ready.connect(self.on_verification)
        task_worker.error_occurred.connect(self.on_task_error)
        view.refresh_requested.connect(self.refresh)
        view.confirm_requested.connect(self.confirm)
        view.debug_requested.connect(self.debug)

    def _set_state(self, state):
        self.state = state
        self.view.set_status(state)

    @Slot()
    def start(self):
        if self._started or self._closing:
            return
        self._started = True
        self.task_thread.start()
        self.cv_thread.start()
        self.refresh()

    @Slot(object)
    def on_frame(self, frame):
        """
        여기서 캡챠 그림 합성
        :param frame:
        :return:
        """
        if self._closing:
            return
        self.latest_frame = frame
        if self.captcha is None:
            self.view.set_captcha(frame)
            return
        height, width = self.captcha.shape[:2]
        left, top, crop_width, crop_height = self._camera_crop_bounds(frame)
        camera = frame[top:top + crop_height, left:left + crop_width]
        # 화면 비율은 crop으로 맞추고, 합성에 필요한 픽셀 크기만 맞춘다.
        if camera.shape[:2] != (height, width):
            camera = cv2.resize(camera, (width, height))

        blended = cv2.addWeighted(
            camera, 0.,
            self.captcha, 1.,
            0,
        )
        self.view.set_captcha(blended)

    def _camera_crop_bounds(self, frame):
        height, width = frame.shape[:2]
        target_height, target_width = self.captcha.shape[:2]
        if width * target_height > height * target_width:
            crop_height = height
            crop_width = max(1, height * target_width // target_height)
        else:
            crop_width = width
            crop_height = max(1, width * target_height // target_width)
        return ((width - crop_width) // 2, (height - crop_height) // 2,
                crop_width, crop_height)

    @Slot(object, object)
    def on_hand(self, position, serial):
        if self._closing:
            return
        # 합성 영상과 동일한 crop/배율을 손 좌표에도 적용한다.
        if position is not None and self.captcha is not None and self.latest_frame is not None:
            left, top, width, height = self._camera_crop_bounds(self.latest_frame)
            x, y = position[0] - left, position[1] - top
            if 0 <= x < width and 0 <= y < height:
                ch, cw = self.captcha.shape[:2]
                position = (x * cw / width, y * ch / height)
            else:
                position = None
        self.position = position
        self.serial = serial
        if position is None:
            self.view.set_hand_position(None, None)
            if self.state == "SOLVING":
                self._set_state("READY")
            return
        if self.state in ("READY", "SOLVING") and self.captcha is not None:
            self._set_state("SOLVING")
            self.view.set_hand_position(*self.position)

    @Slot(str)
    def on_camera_error(self, message):
        if not self._closing:
            print(message)
            self._camera_failed = True
            self._set_state("CAMERA_ERROR")

    @Slot()
    def refresh(self):
        if self._closing or not self._started or self.state in ("GENERATING", "VERIFYING"):
            return
        self._request_id += 1
        self.captcha = None
        self.position = None
        self.serial = None
        self.view.set_hand_position(None, None)
        self._set_state("GENERATING")
        self.generate_requested.emit(self._request_id)

    @Slot(int, object, object, object)
    def on_captcha(self, request_id, image, small, regions):
        if self._closing or request_id != self._request_id:
            return
        self.captcha = image
        self.regions = regions
        self.view.set_reference(small)
        if self.latest_frame is not None:
            self.on_frame(self.latest_frame)
        self._set_state("CAMERA_ERROR" if self._camera_failed else "READY")
        if self.position is not None:
            # 다음 CV 이벤트가 변환된 좌표와 함께 SOLVING으로 전환한다.
            self.position = None

    @Slot()
    def confirm(self):
        if self._closing or self.state != "SOLVING" or self.position is None or self.serial is None:
            return
        x, y = self.position
        self._set_state("VERIFYING")
        self.verify_requested.emit(self._request_id, (x, y, x, y), self.serial.copy())

    @Slot(int, bool, bool)
    def on_verification(self, request_id, correct, is_bot):
        if self._closing or request_id != self._request_id:
            return
        self._set_state("MOCK_SUCCESS" if correct and not is_bot else "MOCK_FAILED")

    @Slot(int, str)
    def on_task_error(self, request_id, message):
        if not self._closing and request_id == self._request_id:
            print(message)
            self._set_state("ERROR")

    @Slot()
    def debug(self):
        print(f"Controller: state={self.state}, hand={self.position}")

    @Slot()
    def shutdown(self):
        if self._closing:
            return
        self._closing = True
        for thread in (self.cv_thread, self.task_thread):
            thread.requestInterruption()
            thread.quit()
        # 실행 중인 호출이 반환하면 종료한다. terminate()로 자원을 강제 파괴하지 않는다.
        for thread in (self.cv_thread, self.task_thread):
            thread.wait()
