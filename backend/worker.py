"""문제 생성, 손 추적, 정답 판정을 직렬 실행하는 Task Worker."""
import cv2
from threading import Lock

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from cv.hand_tracking.hand_tracker import HandTracker
from cv.problem_gen import ProblemGenerator


class TaskWorker(QObject):
    captcha_ready = Signal(int, object, object)
    verification_ready = Signal(int, bool)
    error_occurred = Signal(int, str)
    tracking_updated = Signal(int, object)
    canvas_ready = Signal(int, object)
    _frame_requested = Signal()

    def __init__(self, generator, tracker, parent=None):
        super().__init__(parent)
        self.generator:ProblemGenerator = generator
        self.tracker:HandTracker = tracker
        self._frame_lock = Lock()
        self._latest_frame = None
        self._frame_pending = False
        self._problem_id = None
        self._tracking_request_id = 0
        self._frame_requested.connect(self._process_latest_frame, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def submit_frame(self, frame):
        """카메라 스레드에서 최신 프레임만 보관한다. 인식이나 Qt 객체 조작은 하지 않는다."""
        with self._frame_lock:
            self._latest_frame = frame
            if self._frame_pending:
                return
            self._frame_pending = True
        self._frame_requested.emit()

    @Slot()
    def _process_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        try:
            if frame is None or self._cancelled() or not self.tracker.ready:
                return
            result = self.tracker.process(frame)
            if result.state == 'done':
                x, y = result.stop_position
                box = (x, y, x+100, y+100)
                print(box)
                ret = self.generator.solve(self._problem_id, box)
                if ret['correct']:
                    print('pass | now check integrity')



            if result is not None and not self._cancelled():
                canvas = self.tracker.render(result)
                if canvas is None or self._cancelled():
                    return

                # 카메라를 canvas 비율에 맞춰 중앙 crop한 뒤 같은 크기로 합성한다.
                height, width = canvas.shape[:2]
                frame_height, frame_width = frame.shape[:2]
                if frame_width * height > frame_height * width:
                    crop_height = frame_height
                    crop_width = max(1, frame_height * width // height)
                else:
                    crop_width = frame_width
                    crop_height = max(1, frame_width * height // width)
                left = (frame_width - crop_width) // 2
                top = (frame_height - crop_height) // 2
                camera = frame[top:top + crop_height, left:left + crop_width]
                if camera.shape[:2] != (height, width):
                    camera = cv2.resize(camera, (width, height))
                canvas = cv2.addWeighted(
                    camera, 0.3,  # 카메라 비중
                    canvas, 0.7,  # 캡챠 비중
                    0,
                )
                self.canvas_ready.emit(self._tracking_request_id, canvas)
                self.tracking_updated.emit(self._tracking_request_id, result)
        except Exception as exc:
            self.tracker.set_images(None, None)
            self.error_occurred.emit(self._tracking_request_id, f"손인식 처리 오류: {exc}")
        finally:
            with self._frame_lock:
                again = self._latest_frame is not None and not self._cancelled()
                self._frame_pending = again
                if not again:
                    self._latest_frame = None
            if again:
                self._frame_requested.emit()

    @Slot()
    def close_tracker(self):
        with self._frame_lock:
            self._latest_frame = None
        self.tracker.close()

    def _cancelled(self):
        return QThread.currentThread().isInterruptionRequested()

    @Slot(int)
    def generate(self, request_id):
        if self._cancelled():
            return
        try:
            self._tracking_request_id = request_id
            self.tracker.set_images(None, None)
            self._problem_id = None
            problem_id, image, small = self.generator.create()
            if self._cancelled():
                return
            small = cv2.cvtColor(small, cv2.COLOR_BGRA2BGR)
            if not self._cancelled():
                self._problem_id = problem_id
                self.tracker.set_images(image, small)
                with self._frame_lock:
                    self._latest_frame = None
                self.captcha_ready.emit(request_id, image.copy(), small.copy())
        except Exception as exc:
            self.error_occurred.emit(request_id, f"문제생성 오류: {exc}")

    @Slot(int, object)
    def verify(self, request_id, region):
        if self._cancelled():
            return
        if request_id != self._tracking_request_id or self._problem_id is None:
            return
        try:
            result = self.generator.solve(self._problem_id, region)
            self._problem_id = None
            if self._cancelled():
                return
            self.verification_ready.emit(request_id, bool(result["correct"]))
        except Exception as exc:
            self.error_occurred.emit(request_id, f"정답 판정 오류: {exc}")
