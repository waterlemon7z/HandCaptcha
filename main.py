import sys

from PySide6.QtWidgets import QApplication

from backend.worker import TaskWorker
from cv.camera import Camera
from cv.hand_tracking.hand_tracker import HandTracker
from cv.problem_gen import ProblemGenerator
from gui.controller import Controller
from gui.signal import HandCaptchaUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandCaptchaUI()
    generator = ProblemGenerator()
    tracker = HandTracker()
    camera = Camera()
    task_worker = TaskWorker(generator, tracker)
    controller = Controller(window, camera, task_worker)
    app.aboutToQuit.connect(controller.shutdown)

    window.show()
    controller.start()
    sys.exit(app.exec())
