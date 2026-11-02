import sys

import cv2
from PySide6.QtWidgets import QApplication

from backend.mock import (
    MockBotDetector, MockDifficultyChecker, MockHandRecognizer,
)
from backend.worker import TaskWorker
from cv.camera import Camera
from cv.problem_gen import ProblemGenerator
from gui.controller import Controller
from gui.signal import HandCaptchaUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandCaptchaUI()
    recognizer = MockHandRecognizer()
    generator = ProblemGenerator()
    difficulty_checker = MockDifficultyChecker()
    bot_detector = MockBotDetector()
    camera = Camera(recognizer=recognizer)
    task_worker = TaskWorker(generator, difficulty_checker, bot_detector)
    controller = Controller(window, camera, task_worker)
    app.aboutToQuit.connect(controller.shutdown)

    window.show()
    controller.start()
    sys.exit(app.exec())
