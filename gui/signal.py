import sys

import numpy as np

from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QApplication
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPen, QColor

from gui.handcaptcha import Ui_MainWindow


class HandCaptchaUI(QMainWindow):
    refresh_requested = Signal()
    confirm_requested = Signal()
    debug_requested = Signal()

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scene = QGraphicsScene(self)
        self.ui.bigImg.setScene(self.scene)
        self.image_item = self.scene.addPixmap(QPixmap())

        self.reference_scene = QGraphicsScene(self)
        self.ui.smallImg.setScene(self.reference_scene)
        self.reference_item = self.reference_scene.addPixmap(QPixmap())
        self.hand_item = self.scene.addEllipse(-6, -6, 12, 12, QPen(QColor("#16815E"), 3))
        self.hand_item.setZValue(1)
        self.hand_item.hide()

        self.connect_events()

    def connect_events(self):
        self.ui.refreshButton.clicked.connect(self.refresh_requested.emit)
        self.ui.confirm.clicked.connect(self.confirm_requested.emit)
        self.ui.debugButton.clicked.connect(self.debug_requested.emit)

    @Slot(object)
    def set_captcha(self, image: np.ndarray):
        """uint8 회색조 또는 OpenCV BGR/BGRA 이미지를 표시한다."""
        if image.dtype != np.uint8 or image.size == 0:
            raise ValueError("image must be a non-empty uint8 array")
        if image.ndim == 2:
            image_format = QImage.Format.Format_Grayscale8
        elif image.ndim == 3 and image.shape[2] == 3:
            image_format = QImage.Format.Format_BGR888
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, [2, 1, 0, 3]]
            image_format = QImage.Format.Format_RGBA8888
        else:
            raise ValueError("image must have shape (H, W), (H, W, 3), or (H, W, 4)")

        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data, width, height, image.strides[0], image_format
        ).copy()
        self.image_item.setPixmap(QPixmap.fromImage(qimage))
        self.scene.setSceneRect(0, 0, width, height)
        self.ui.bigImg.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    @Slot(str)
    def set_status(self, text: str):
        self.ui.statusValue.setText(text)

    @Slot(object, object)
    def set_hand_position(self, x, y):
        if x is None or y is None:
            self.hand_item.hide()
        else:
            self.hand_item.setPos(float(x), float(y))
            self.hand_item.show()

    @Slot(object)
    def set_reference(self, image):
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        qimage = QImage(image.data, width, height, image.strides[0],
                        QImage.Format.Format_BGR888).copy()
        self.reference_item.setPixmap(QPixmap.fromImage(qimage))
        self.reference_scene.setSceneRect(0, 0, width, height)
        self.ui.smallImg.fitInView(self.reference_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.scene.sceneRect().isEmpty():
            self.ui.bigImg.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = HandCaptchaUI()
    window.show()

    sys.exit(app.exec())
