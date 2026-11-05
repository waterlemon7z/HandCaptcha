# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'handcaptcha.ui'
##
## Created by: Qt User Interface Compiler version 6.12.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGraphicsView, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(700, 750)
        MainWindow.setMinimumSize(QSize(700, 750))
        MainWindow.setMaximumSize(QSize(700, 750))
        MainWindow.setStyleSheet(u"QWidget#centralwidget {\n"
"    background-color: #ECEFF3;\n"
"}\n"
"QFrame#captchaCard {\n"
"    background-color: #FFFFFF;\n"
"    border: 1px solid #D5DCE4;\n"
"    border-radius: 8px;\n"
"}\n"
"QFrame#headerFrame {\n"
"    background-color: #478FDF;\n"
"    border: none;\n"
"    border-radius: 4px;\n"
"}\n"
"QLabel#HandCAPTCHA_title {\n"
"    background: transparent;\n"
"    color: #FFFFFF;\n"
"    font-size: 30px;\n"
"    font-weight: 800;\n"
"}\n"
"QLabel#how2captcha {\n"
"    background: transparent;\n"
"    color: #F4F8FF;\n"
"    font-size: 15px;\n"
"}\n"
"QGraphicsView#bigImg {\n"
"    background-color: #EEF2F7;\n"
"    border: 1px solid #E2E8F0;\n"
"    border-radius: 2px;\n"
"}\n"
"QWidget#footerWidget {\n"
"    background-color: #FFFFFF;\n"
"}\n"
"QPushButton#refreshButton, QPushButton#infoButton {\n"
"    background-color: transparent;\n"
"    color: #667587;\n"
"    border: 1px solid transparent;\n"
"    border-radius: 5px;\n"
"    font-size: 23px;\n"
"    padding: 2px 6px;\n"
"}\n"
"QComboBox#verificationCombo {\n"
"    color: #667587;\n"
"    background-color: transparent;\n"
"    border: 1px solid #D5DCE4;\n"
"    border-radius: 5px;\n"
"    font-size: 13px;\n"
"    padding: 6px;\n"
"}\n"
"QPushButton#refreshButton:hover, QPushButton#infoButton:hover {\n"
"    background-color: #EDF2F8;\n"
"    border-color: #E2E8F0;\n"
"}\n"
"QLabel#statueLabel {\n"
"    color: #687483;\n"
"    font-size: 12px;\n"
"}\n"
"QLabel#statusValue {\n"
"    color: #16815E;\n"
"    font-size: 12px;\n"
"    font-weight: bold;\n"
"}\n"
"QPushButton#confirm {\n"
"    background-color: #478FDF;\n"
"    color: #FFFFFF;\n"
"    border: none;\n"
"    border-radius: 5px;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    padding: 10px 24px;\n"
"}\n"
"QPushButton#confirm:hover {\n"
"    background-color: #367DC9;\n"
"}\n"
"QPushButton#confirm:pressed {\n"
"    background-color: #276BB5;\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.rootLayout = QVBoxLayout(self.centralwidget)
        self.rootLayout.setSpacing(0)
        self.rootLayout.setObjectName(u"rootLayout")
        self.rootLayout.setContentsMargins(20, 20, 20, 20)
        self.captchaCard = QFrame(self.centralwidget)
        self.captchaCard.setObjectName(u"captchaCard")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.captchaCard.sizePolicy().hasHeightForWidth())
        self.captchaCard.setSizePolicy(sizePolicy)
        self.captchaCard.setMaximumSize(QSize(1100, 16777215))
        self.captchaCard.setFrameShape(QFrame.Shape.StyledPanel)
        self.cardLayout = QVBoxLayout(self.captchaCard)
        self.cardLayout.setSpacing(10)
        self.cardLayout.setObjectName(u"cardLayout")
        self.cardLayout.setContentsMargins(10, 10, 10, 10)
        self.headerFrame = QFrame(self.captchaCard)
        self.headerFrame.setObjectName(u"headerFrame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.headerFrame.sizePolicy().hasHeightForWidth())
        self.headerFrame.setSizePolicy(sizePolicy1)
        self.headerFrame.setMinimumSize(QSize(0, 132))
        self.headerFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.horizontalLayout_2 = QHBoxLayout(self.headerFrame)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(25, 18, 25, 18)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.HandCAPTCHA_title = QLabel(self.headerFrame)
        self.HandCAPTCHA_title.setObjectName(u"HandCAPTCHA_title")
        self.HandCAPTCHA_title.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.verticalLayout.addWidget(self.HandCAPTCHA_title)

        self.how2captcha = QLabel(self.headerFrame)
        self.how2captcha.setObjectName(u"how2captcha")
        self.how2captcha.setWordWrap(True)

        self.verticalLayout.addWidget(self.how2captcha)


        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.smallImg = QGraphicsView(self.headerFrame)
        self.smallImg.setObjectName(u"smallImg")
        self.smallImg.setMaximumSize(QSize(128, 128))

        self.horizontalLayout_2.addWidget(self.smallImg)


        self.cardLayout.addWidget(self.headerFrame)

        self.bigImg = QGraphicsView(self.captchaCard)
        self.bigImg.setObjectName(u"bigImg")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(1)
        sizePolicy2.setHeightForWidth(self.bigImg.sizePolicy().hasHeightForWidth())
        self.bigImg.setSizePolicy(sizePolicy2)
        self.bigImg.setMinimumSize(QSize(480, 270))
        self.bigImg.setMaximumSize(QSize(640, 480))
        self.bigImg.setFrameShape(QFrame.Shape.NoFrame)
        self.bigImg.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.bigImg.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.cardLayout.addWidget(self.bigImg)

        self.footerWidget = QWidget(self.captchaCard)
        self.footerWidget.setObjectName(u"footerWidget")
        sizePolicy1.setHeightForWidth(self.footerWidget.sizePolicy().hasHeightForWidth())
        self.footerWidget.setSizePolicy(sizePolicy1)
        self.footerWidget.setMinimumSize(QSize(0, 62))
        self.footerLayout = QHBoxLayout(self.footerWidget)
        self.footerLayout.setSpacing(8)
        self.footerLayout.setObjectName(u"footerLayout")
        self.footerLayout.setContentsMargins(4, 4, 4, 4)
        self.refreshButton = QPushButton(self.footerWidget)
        self.refreshButton.setObjectName(u"refreshButton")
        self.refreshButton.setMinimumSize(QSize(44, 44))

        self.footerLayout.addWidget(self.refreshButton)

        self.infoButton = QPushButton(self.footerWidget)
        self.infoButton.setObjectName(u"infoButton")
        self.infoButton.setMinimumSize(QSize(44, 44))

        self.footerLayout.addWidget(self.infoButton)

        self.verificationCombo = QComboBox(self.footerWidget)
        self.verificationCombo.setObjectName(u"verificationCombo")
        self.verificationCombo.setMinimumSize(QSize(120, 36))
        self.verificationCombo.addItem("ORB/SIFT", "auto")
        self.verificationCombo.addItem("ORB", "orb")
        self.verificationCombo.addItem("SIFT", "sift")
        self.verificationCombo.setAccessibleName("검증 방식")

        self.footerLayout.addWidget(self.verificationCombo)

        self.footerSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.footerLayout.addItem(self.footerSpacer)

        self.statueLabel = QLabel(self.footerWidget)
        self.statueLabel.setObjectName(u"statueLabel")

        self.footerLayout.addWidget(self.statueLabel)

        self.statusValue = QLabel(self.footerWidget)
        self.statusValue.setObjectName(u"statusValue")

        self.footerLayout.addWidget(self.statusValue)

        self.confirm = QPushButton(self.footerWidget)
        self.confirm.setObjectName(u"confirm")
        self.confirm.setMinimumSize(QSize(115, 44))

        self.footerLayout.addWidget(self.confirm)


        self.cardLayout.addWidget(self.footerWidget)


        self.rootLayout.addWidget(self.captchaCard)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"HandCAPTCHA", None))
        self.HandCAPTCHA_title.setText(QCoreApplication.translate("MainWindow", u"HandCAPTCHA", None))
        self.how2captcha.setText(QCoreApplication.translate("MainWindow", u"\uc190\uc744 \uc2dc\uc791\uc601\uc5ed\uc5d0 \ube44\ucd94\uace0 \ub300\uae30 \ud6c4 \uc2dc\uc791\ud558\uc138\uc694. \n"
"\uc624\ub978\ucabd \uadf8\ub9bc\uc774 \uc788\ub294 \uacf3\uc5d0 \uc190\uc744 \ube44\ucd94\uace0 \uba48\ucdb0\uc11c \uc81c\ucd9c\ud558\uc138\uc694.", None))
#if QT_CONFIG(tooltip)
        self.bigImg.setToolTip(QCoreApplication.translate("MainWindow", u"1280\u00d7720 CAPTCHA \uc774\ubbf8\uc9c0 \ud45c\uc2dc \uc601\uc5ed", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.refreshButton.setToolTip(QCoreApplication.translate("MainWindow", u"\ub2e4\ub978 CAPTCHA", None))
#endif // QT_CONFIG(tooltip)
        self.refreshButton.setText(QCoreApplication.translate("MainWindow", u"\u21bb", None))
#if QT_CONFIG(tooltip)
        self.infoButton.setToolTip(QCoreApplication.translate("MainWindow", u"\uc0ac\uc6a9 \ubc29\ubc95", None))
#endif // QT_CONFIG(tooltip)
        self.infoButton.setText(QCoreApplication.translate("MainWindow", u"\u24d8", None))
        self.verificationCombo.setToolTip("캡챠 자동 풀이 검증 방식. 다음 문제 생성부터 적용됩니다.")
        self.statueLabel.setText(QCoreApplication.translate("MainWindow", u"Status:", None))
        self.statusValue.setText(QCoreApplication.translate("MainWindow", u"READY", None))
        self.confirm.setText(QCoreApplication.translate("MainWindow", u"\ud655\uc778", None))
    # retranslateUi
