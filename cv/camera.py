"""CV 스레드에서 카메라 프레임을 읽어 전달한다."""

import cv2
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot


class Camera(QObject):
    frame_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, device_index=0, width=1280, height=720, parent=None,
                 capture_factory=None):
        super().__init__(parent)
        self.device_index = device_index
        self.width = width
        self.height = height
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._capture = None
        self._timer = None

    @Slot()
    def start(self):
        """Controller가 CV 스레드 시작 시 호출한다."""
        if self._capture is not None:
            return
        try:
            self._capture = self._capture_factory(self.device_index)
            if not self._capture.isOpened():
                raise RuntimeError(f"카메라 {self.device_index}를 열 수 없습니다. 장치와 접근 권한을 확인하세요.")
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._read_frame)
            self._timer.start(33)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.stop()

    @Slot()
    def _read_frame(self):
        try:
            if QThread.currentThread().isInterruptionRequested():
                self.stop()
                return
            success, frame = self._capture.read()
            if QThread.currentThread().isInterruptionRequested():
                self.stop()
                return
            if not success or frame is None:
                raise RuntimeError("카메라 프레임을 읽을 수 없습니다.")
            frame = cv2.flip(frame, 1)
            self.frame_ready.emit(frame.copy())
        except Exception as exc:
            self.error_occurred.emit(f"카메라 처리 오류: {exc}")
            self.stop()

    @Slot()
    def stop(self):
        """반드시 CV 스레드에서 실행하여 타이머와 카메라를 해제한다."""
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.finished.emit()
