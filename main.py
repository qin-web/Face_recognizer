import sys
import os
import cv2
import threading
from PyQt5.QtWidgets import QApplication ,QMainWindow,QMessageBox,QFileDialog
from PyQt5.QtCore import QBasicTimer,pyqtSignal,Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import *
from face_ui import Mainwindow
from realtime_capture import *
from classification import *

class MyMainWindow(QMainWindow, Mainwindow):
    signal = pyqtSignal()
    def __init__(self, parent = None):
        super(MyMainWindow,self).__init__(parent)
        self.setup(self)
        self.source = 0
        self.cap = cv2.VideoCapture()
        self.video_btn = 0 #区别打开摄像头和人脸识别
        
        self.pushButton_camera.clicked.connect(self.open_camera)#打开摄像头
        self.pushButton_recognition.clicked.connect(self.recognition)#人脸识别
        self.pushButton_photo.clicked.connect(self.take_photo)#拍照
        self.show()

    def open_camera(self):
        pass
    
    def recognition(self):
        realtime_recognition()

    def take_photo(self):
        pass

    def train_data(self):
        train()

if __name__ == "__main__":
    app =  QApplication(sys.argv)
    myWin = MyMainWindow()
    # myWin.signal.connect()
    sys.exit(app.exec_())