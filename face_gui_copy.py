import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QMessageBox,
    QHBoxLayout, QVBoxLayout, QApplication, QInputDialog, QLineEdit)
from PyQt5.QtCore import QTimer, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QFontDatabase
from recognition import *
from classification import *
from resize_face import *
import cv2
from photo import *
from token_photo import *
from Image_Dataset_Generator import *
#姿态估计
from fuc_yaw import calculate
# 检测
import os
import facenet
import tensorflow as tf
import numpy as np
import detect_face
import pickle
import matplotlib.pyplot as plt

import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner

import glob
from scipy.spatial import distance
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *

import shutil
import time


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.timer = QTimer()

        self.timer_re = QTimer()
        self.timer_yaw = QTimer()
        self.name = 'a'

        self.initUI()

    def initUI(self):
        self.Main_layout = QHBoxLayout()
        self.Button_layout = QVBoxLayout()

        #设置按钮
        self.PhotoButton = QPushButton('拍照')
        self.OpenCameraButton = QPushButton('输入用户信息')
        self.RecognitionButton = QPushButton('人脸识别')
        self.CheckinButton = QPushButton('录入数据')
        self.TrainButton = QPushButton('训练数据')
        self.QuitButton = QPushButton('退出')
        self.YawButton = QPushButton('人脸姿态评估')

        # self.PhotoButton.setFont(QFont("Roman times",10,QFont.Bold))
        # self.OpenCameraButton.setFont(QFont("Roman times",10,QFont.Bold))
        # self.RecognitionButton.setFont(QFont("Roman times",10,QFont.Bold))
        # self.CheckinButton.setFont(QFont("Roman times",10,QFont.Bold))
        # self.TrainButton.setFont(QFont("Roman times",10,QFont.Bold))
        # self.QuitButton.setFont(QFont("Roman times",10,QFont.Bold))

        self.CameraLabel = QLabel()
        self.CameraLabel.setFixedSize(641,481)

        self.MoveLabel = QLabel()
        self.MoveLabel.setFixedSize(100, 200)
        
        self.Button_layout.addWidget(self.PhotoButton)
        self.Button_layout.addWidget(self.OpenCameraButton)
        self.Button_layout.addWidget(self.RecognitionButton)
        self.Button_layout.addWidget(self.YawButton)
        self.Button_layout.addWidget(self.CheckinButton)
        self.Button_layout.addWidget(self.TrainButton)
        self.Button_layout.addWidget(self.QuitButton)
        self.Button_layout.addWidget(self.MoveLabel)

        self.Main_layout.addLayout(self.Button_layout)
        self.Main_layout.addWidget(self.CameraLabel)

        # 打开摄像头响应事件
        self.OpenCameraButton.clicked.connect(self.button_open_camera_clicked)
        # self.timer.timeout.connect(self.play_video)#若定时器结束，则调用play_video()
        #人脸识别响应事件
        self.RecognitionButton.clicked.connect(self.recognition_button_clicked)
        self.timer_re.timeout.connect(self.video_recognition)
        #人脸姿态估计
        self.YawButton.clicked.connect(self.yaw_button_clicked)
        self.timer_yaw.timeout.connect(self.video_yaw)
        #拍照响应事件
        self.PhotoButton.clicked.connect(self.take_picture)
        # #录入数据响应事件
        # self.CheckinButton.clicked.connect()
        #训练数据响应事件
        self.TrainButton.clicked.connect(self.train_data)
        #退出
        self.QuitButton.clicked.connect(self.close)

        self.setLayout(self.Main_layout)

        self.setGeometry(300, 300, 800, 500)
        self.setWindowTitle('人脸识别')
        self.show()

    def take_picture(self):
        # if self.timer.isActive() == True:
        number_of_images = take_data(self.name)
        message = QMessageBox.information(self, '提示', '一共拍摄'+ str(number_of_images) + "张", QMessageBox.Ok)

    def button_open_camera_clicked(self):
        # print(self.timer.isActive())
        # if self.timer.isActive() == False: #计时器没有启动
        #     flag = self.cap.open(self.CAM_NUM)
        #     if flag == False:
        #         msg = QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QMessageBox.Ok)
        #     else:
        self.name, ok = QInputDialog.getText(self, "输入用户名", "输入用户名", QLineEdit.Normal, "")
                # self.timer.start(30)#每30毫秒取一帧
                # self.OpenCameraButton.setText('关闭摄像头')

        # else:
        #     self.timer.stop()
        #     self.cap.release()
        #     self.CameraLabel.clear()
        #     self.OpenCameraButton.setText('打开摄像头')

    def play_video(self):#从视频流中读图片放入label
        flag, image = self.cap.read()
        if flag:
            image = cv2.resize(image, (640, 480))
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.CameraLabel.setPixmap(QPixmap.fromImage(showImage))


    def recognition_button_clicked(self):
        # print(self.timer_re.isActive())
        if self.timer_re.isActive() == False:
            # print('in')
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QMessageBox.Ok)
            else:
                self.timer_re.start(30)
                self.RecognitionButton.setText('关闭人脸识别')

                self.FR_model = load_model('nn4.small2.v1.h5')
                # print("Total Params:", FR_model.count_params())

                self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

                self.threshold = 0.25

                self.face_database = {}

                for name in os.listdir('images'):
                    for image in os.listdir(os.path.join('images',name)):
                        identity = os.path.splitext(os.path.basename(image))[0]
                        self.face_database[identity] = fr_utils.img_path_to_encoding(os.path.join('images',name,image), self.FR_model)

                self.minsize = 20
                self.threshold1 = [0.6, 0.7, 0.7]
                self.factor = 0.709
                self.image_size = 160
                with tf.Graph().as_default():
                    self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
                    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
                    with self.sess.as_default():
                        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
                        facenet.load_model('models')

                        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                        self.embedding_size = self.embeddings.get_shape()[1]

                        self.classifier_filename_exp = os.path.expanduser('new_models.pkl')
                        with open(self.classifier_filename_exp, 'rb') as infile:
                            (self.model, self.class_names) = pickle.load(infile)
        else:
            self.timer_re.stop()
            self.cap.release()
            self.CameraLabel.clear()
            self.RecognitionButton.setText('人脸识别')
            self.sess.close()

    def video_recognition(self):
        flag, frame = self.cap.read()
        if flag:
            frame = cv2.flip(frame, 1)

            # faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
            # for(x,y,w,h) in faces:
            #     # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #     roi = frame[y:y+h, x:x+w]
                
                    # print('Min dist: ',min_dist)

                # if min_dist < 0.1:
                #     cv2.putText(frame, "Face : " + identity[:-1], (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                #     # cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                # else:
                #     cv2.putText(frame, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.ndim == 2:
                gray = facenet.to_rgb(gray)

                bounding_boxes, points = detect_face.detect_face(gray, self.minsize, self.pnet, self.rnet, self.onet ,self.threshold1, self.factor)
                # nrof_faces = bounding_boxes.shape[0]

            for face_position in bounding_boxes:
                face_position = face_position.astype(int)

                cropped = gray[face_position[1]-40:face_position[3]+40,face_position[0]-40:face_position[2]+40,:]

                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    continue

                scaled = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                encoding = img_to_encoding(scaled, self.FR_model)
                min_dist = 100
                identity = None

                for(name, encoded_image_name) in self.face_database.items():
                    dist = np.linalg.norm(encoding - encoded_image_name)
                    if(dist < min_dist):
                        min_dist = dist
                        identity = name
                plt.imshow(scaled)
                scaled = scaled.reshape(-1, self.image_size, self.image_size, 3)

                emb_array = self.sess.run(self.embeddings, feed_dict={self.images_placeholder: scaled, self.phase_train_placeholder: False})

                # print(emb_array)
                predict = self.model.predict(emb_array)
                # print(predict)
                
                cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (255, 255, 0), 2)
                person_name = ''.join([i for i in identity[:-1] if not i.isdigit()])
                cv2.putText(frame, person_name , (face_position[0], face_position[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), thickness=2, lineType=2)

                if points.shape[0] != 0:
                    for i in range(points.shape[1]):
                        count = points.shape[0]/2
                        count = int(count)
                        for j in range(count):
                            cv2.circle(frame, (points[j][i], points[j + count][i]), 3, (255, 255, 0), -1)
            frame = cv2.resize(frame, (640, 480))
            # image = realtime_recognition(image)
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.CameraLabel.setPixmap(QPixmap.fromImage(showImage))

    def yaw_button_clicked(self):
        if self.timer_yaw.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QMessageBox.Ok)
            else:
                self.timer_yaw.start(30)#每30毫秒取一帧
                self.YawButton.setText('关闭姿态估计')

        else:
            self.timer_yaw.stop()
            self.cap.release()
            self.CameraLabel.clear()
            self.YawButton.setText('人脸姿态估计')

    def video_yaw(self):
        flag, image = self.cap.read()
        if flag:
            image = calculate(image)
            image = cv2.resize(image, (640, 480))
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.CameraLabel.setPixmap(QPixmap.fromImage(showImage))
    
    def train_data(self):
        # train()
        # os.remove('/home/wjc/Documents/code/face_test/images/undefine')
        shutil.rmtree('/home/wjc/Documents/code/face_test/images/undefine')
        time.sleep(5)
        message = message = QMessageBox.information(self, '提示', '训练完成', QMessageBox.Ok)

# class ShowPhotoWindow(QWidget):
#     def __init__(self):
#         super().__init__()

#         self.initUI()
        

#     def initUI(self):
#         self.Main_layout = QVBoxLayout()
#         self.Button_layout = QHBoxLayout()
        
#         self.PhotoLabel = QLabel()
#         self.PhotoLabel.setFixedSize(160, 160)

#         self.SaveButton = QPushButton('保存图片')
#         self.RetakeButton = QPushButton('重新拍照')

#         self.Button_layout.addWidget(self.SaveButton)
#         self.Button_layout.addWidget(self.RetakeButton)

#         self.Main_layout.addWidget(self.PhotoLabel)
#         self.Main_layout.addLayout(self.Button_layout)

#         self.SaveButton.clicked.connect(self.save)
#         self.RetakeButton.clicked.connect(self.exit)

#         self.setLayout(self.Main_layout)
        
#         self.setGeometry(500, 500, 300, 300)
#         self.setWindowTitle('照片')
#         # self.show()
#         # print('in')

#     def save(self):
#         pass

#     def exit(self):
#         pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    loadedFontID = QFontDatabase.addApplicationFont("./msyh.ttf")
    loadedFontFamilies = QFontDatabase.applicationFontFamilies(loadedFontID)
    if(list(loadedFontFamilies).__len__()>0):
        fontName = loadedFontFamilies[0]
        font  = QFont(fontName)
        app.setFont(font)

    window = MainWindow()
    childwindow = win()
    btn = window.CheckinButton
    btn.clicked.connect(childwindow.show)
    # tokenwindow = win_1()
    # btn_1 =window.PhotoButton
    # btn_1.clicked.connect(tokenwindow.show)
    sys.exit(app.exec_())