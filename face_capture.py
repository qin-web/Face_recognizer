import cv2
import sys
import os
import tensorflow as tf
import numpy as np
import detect_face
import facenet

video_capture = cv2.VideoCapture(0)
capture_interval = 1
capture_num = 100
capture_count = 0
frame_count = 0
detect_multiple_faces = False

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

while True:
    ret, frame = video_capture.read()
    # print('t')
    if(capture_count % capture_interval == 0):
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        for face_position in bounding_boxes:
            face_position = face_position.astype(int)
            cropped = frame[face_position[1] : face_position[3], face_position[0] : face_position[2], :]
            scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('/home/wjc/Documents/code/hwg' + str(frame_count) + '.jpg', scaled)

        frame_count += 1

    capture_count += 1

    if frame_count >= capture_num:
        break
    
video_capture.release()
cv2.destroyAllWindows()
print('Capture Over')
