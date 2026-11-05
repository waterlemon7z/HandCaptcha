"""
Author  : MinSeok Choi
File    : controller.py
Role    : GUI의 버튼 연동 및 상태 변화에 따른 업데이트
"""
import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

class Controller(QObject):
    generate_requested = Signal(int, str)

    def __init__(self, view, camera, task_worker):
        super().__init__()

        # 화면과 캡챠 상태
        self.view = view
        self.position = None
        self.captcha = None # 실제 captcha 이미지
        self.state = "READY"
        self._request_id = 0

        # 실행 및 종료 상태
        self._started = False
        self._closing = False
        self._camera_failed = False

        # 작업별 스레드 생성 및 객체 배치
        self.cv_thread = QThread(self)
        self.cv_thread.setObjectName("CV Worker")
        self.task_thread = QThread(self)
        self.task_thread.setObjectName("Task Worker")
        camera.moveToThread(self.cv_thread)
        task_worker.moveToThread(self.task_thread)

        # 스레드 시작과 종료 시 자원 정리
        self.cv_thread.started.connect(camera.start)
        # finished는 해당 작업 스레드에서 발생한다. 이벤트 루프가 끝나도 해제한다.
        self.cv_thread.finished.connect(camera.stop, Qt.ConnectionType.DirectConnection)
        self.cv_thread.finished.connect(camera.deleteLater)
        self.task_thread.finished.connect(task_worker.deleteLater)
        self.task_thread.finished.connect(task_worker.close_tracker, Qt.ConnectionType.DirectConnection)
        camera.finished.connect(self.cv_thread.quit, Qt.ConnectionType.DirectConnection)

        # Controller → TaskWorker: 문제 생성 요청
        self.generate_requested.connect(task_worker.generate_with_method, Qt.ConnectionType.QueuedConnection)

        # Camera → GUI / TaskWorker: 화면 표시 및 최신 프레임 전달
        camera.frame_ready.connect(self.on_frame, Qt.ConnectionType.QueuedConnection)
        # submit_frame은 잠금으로 프레임만 교체한다. 인식은 TaskWorker 스레드에서 실행한다.
        camera.frame_ready.connect(task_worker.submit_frame, Qt.ConnectionType.DirectConnection)

        # TaskWorker → Controller: 인식 결과와 화면, 캡챠 및 판정 결과
        task_worker.tracking_updated.connect(self.on_tracking_result, Qt.ConnectionType.QueuedConnection)
        task_worker.canvas_ready.connect(self.on_canvas, Qt.ConnectionType.QueuedConnection)
        task_worker.captcha_ready.connect(self.on_captcha)
        task_worker.verification_ready.connect(self.on_verification)

        # 오류 처리
        camera.error_occurred.connect(self.on_camera_error)
        task_worker.error_occurred.connect(self.on_task_error)

        # GUI → Controller: 버튼 이벤트
        view.refresh_requested.connect(self.refresh)
        view.confirm_requested.connect(self.confirm)

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
        """캡챠 준비 전에는 검은 배경에 대기 문구를 표시한다."""
        if self._closing:
            return
        if self.captcha is None:
            height, width = frame.shape[:2]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            text = "wait.."
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(text, font, 1.0, 2)
            origin = ((width - text_width) // 2, (height + text_height - baseline) // 2)
            cv2.putText(canvas, text, origin, font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            self.view.set_captcha(canvas)

    @Slot(int, object)
    def on_canvas(self, request_id, canvas):
        """현재 캡챠에 대해 tracker가 완성한 화면을 표시한다."""
        if (self._closing or request_id != self._request_id or self.captcha is None
                or self.state in ("ANSWER_CORRECT", "ANSWER_INCORRECT")):
            return
        self.view.set_captcha(canvas)

    @Slot(int, object)
    def on_tracking_result(self, request_id, result):
        if (self._closing or request_id != self._request_id or self.captcha is None
                or self.state in ("ANSWER_CORRECT", "ANSWER_INCORRECT")):
            return
        # tracker 결과는 이미 문제 이미지 좌표이므로 카메라 crop 변환을 하지 않는다.
        self.position = result.position
        if self.position is None:
            self.view.set_hand_position(None, None)
            if self.state == "SOLVING":
                self._set_state("READY")
        elif self.state in ("READY", "SOLVING"):
            self._set_state("SOLVING")
            # 손 위치는 tracker.render()에서 canvas에 그린다.

    @Slot(str)
    def on_camera_error(self, message):
        if not self._closing:
            print(message)
            self._camera_failed = True
            if self.state not in ("ANSWER_CORRECT", "ANSWER_INCORRECT"):
                self._set_state("CAMERA_ERROR")

    @Slot()
    def refresh(self):
        if (self._closing or not self._started
                or self.state in ("GENERATING", "ANSWER_CORRECT", "ANSWER_INCORRECT")):
            return
        self._request_id += 1
        self.captcha = None
        self.position = None
        self.view.set_hand_position(None, None)
        self._set_state("GENERATING")
        self.generate_requested.emit(self._request_id, self.view.verification_method())

    @Slot(int, object, object)
    def on_captcha(self, request_id, image, small):
        if self._closing or request_id != self._request_id:
            return
        self.captcha = image
        self.view.set_reference(small)
        self.view.set_captcha(image)
        self._set_state("CAMERA_ERROR" if self._camera_failed else "READY")

    @Slot()
    def confirm(self):
        if self._closing or self.state not in ("ANSWER_CORRECT", "ANSWER_INCORRECT"):
            return
        self._set_state("READY")
        self.refresh()

    @Slot(int, bool)
    def on_verification(self, request_id, correct):
        if self._closing or request_id != self._request_id:
            return
        self._set_state("ANSWER_CORRECT" if correct else "ANSWER_INCORRECT")
        self.view.set_status(
            "정답입니다. confirm 버튼을 누르면 다시 시작합니다." if correct
            else "오답입니다. confirm 버튼을 누르면 다시 시작합니다."
        )

    @Slot(int, str)
    def on_task_error(self, request_id, message):
        if not self._closing and request_id == self._request_id:
            print(message)
            self._set_state("ERROR")

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
