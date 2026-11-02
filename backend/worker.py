"""문제생성 → 난이도검수 및 정답/봇 검사를 직렬 실행하는 Task Worker."""

from PySide6.QtCore import QObject, QThread, Signal, Slot


class TaskWorker(QObject):
    captcha_ready = Signal(int, object, object, object)
    verification_ready = Signal(int, bool, bool)
    error_occurred = Signal(int, str)

    def __init__(self, generator, difficulty_checker, bot_detector, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.difficulty_checker = difficulty_checker
        self.bot_detector = bot_detector

    def _cancelled(self):
        return QThread.currentThread().isInterruptionRequested()

    @Slot(int)
    def generate(self, request_id):
        if self._cancelled():
            return
        try:
            image = self.generator.requestCaptcha((480, 240, 800, 480))
            if self._cancelled():
                return
            small = self.generator.small_image
            regions = self.difficulty_checker.inspect(image, small)
            if not self._cancelled():
                self.captcha_ready.emit(request_id, image.copy(), small.copy(), regions)
        except Exception as exc:
            self.error_occurred.emit(request_id, f"문제생성/난이도검수 오류: {exc}")

    @Slot(int, object, object)
    def verify(self, request_id, region, serial):
        if self._cancelled():
            return
        try:
            correct = self.generator.isAnswer(region)
            if self._cancelled():
                return
            is_bot = self.bot_detector.is_bot(serial)
            if not self._cancelled():
                self.verification_ready.emit(request_id, bool(correct), bool(is_bot))
        except Exception as exc:
            self.error_occurred.emit(request_id, f"정답/봇검출 오류: {exc}")
