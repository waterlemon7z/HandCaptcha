import argparse
import functools
import sys
from pathlib import Path

import cv2
from PySide6.QtWidgets import QApplication

from backend.worker import TaskWorker
from cv.camera import Camera
from cv.hand_tracking.hand_tracker import HandTracker
from cv.problem_gen import ProblemGenerator
from gui.controller import Controller
from gui.signal import HandCaptchaUI
from cv.verification.difficulty_check import DifficultyCheck
from logger import INFO

# 봇 시연 영상 폴더 (레포 바깥). backend/bot_detection/demo_bot.py가 여기에 bot_<seed>.avi로 저장한다
BOT_VIDEO_DIR = Path(__file__).resolve().parent.parent


class LoopVideo:
    """영상 파일을 웹캠처럼 끝없이 반복 재생한다 (봇 시연용)."""

    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)

    def isOpened(self):
        return self._cap.isOpened()

    def set(self, *args):
        return True

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return ok, frame

    def release(self):
        self._cap.release()


def parse_args():
    """
    python main.py                → 웹캠, 랜덤 문제 (평소)
    python main.py --seed 1234    → 웹캠, 문제 고정
    python main.py --bot 1234     → 봇 시연: 문제 1234 고정 + 웹캠 대신 bot_1234.avi 반복 재생
    python main.py --camera 1     → 1번 카메라 (영상 파일 경로도 가능)
    """
    parser = argparse.ArgumentParser(description="HandCaptcha")
    parser.add_argument("--seed", type=int, help="모든 문제를 이 시드로 만든다")
    parser.add_argument("--bot", type=int, metavar="SEED", help="봇 시연: 이 시드 문제 + bot_<SEED>.avi 영상")
    parser.add_argument("--camera", help="카메라 번호 또는 영상 파일 경로 (기본 0)")
    args, qt_args = parser.parse_known_args()
    if args.bot is not None:
        args.seed = args.bot
        args.camera = args.camera or str(BOT_VIDEO_DIR / f"bot_{args.bot}.avi")
    return args, qt_args


def make_camera(source):
    if source is None:
        return Camera()
    if source.isdigit():
        return Camera(device_index=int(source))
    if not Path(source).exists():
        sys.exit(f"영상 파일이 없습니다: {source}\n먼저 python -m backend.bot_detection.demo_bot --seed <시드> 로 만드세요")
    INFO(f'봇 시연: 웹캠 대신 영상 재생 ({source})')
    return Camera(device_index=source, capture_factory=LoopVideo)


if __name__ == "__main__":
    args, qt_args = parse_args()
    app = QApplication([sys.argv[0], *qt_args])
    INFO('Program started')
    dif = DifficultyCheck()
    window = HandCaptchaUI()
    generator = ProblemGenerator(validator=dif)
    if args.seed is not None:
        # 시연용: 모든 문제를 같은 시드로 만든다 (봇 영상이 이 문제의 정답으로 감)
        INFO(f'문제 시드 고정: {args.seed}')
        generator.create = functools.partial(generator.create, seed=args.seed)
    tracker = HandTracker()
    camera = make_camera(args.camera)
    task_worker = TaskWorker(generator, tracker)
    controller = Controller(window, camera, task_worker)
    app.aboutToQuit.connect(controller.shutdown)

    window.show()
    controller.start()
    sys.exit(app.exec())
