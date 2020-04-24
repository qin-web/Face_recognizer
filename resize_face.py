import cv2
import sys
import os
import tensorflow as tf
import numpy as np
import detect_face
import facenet

def cut_picture(image):
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
            # nrof_faces = bounding_boxes.shape[0]

            for face_position in bounding_boxes:
                face_position = face_position.astype(int)
                # print(face_position[1])
                # print(face_position[3])
                # print(face_position[0])
                # print(face_position[2])
                cropped = image[face_position[1]-40 : face_position[3]+40, face_position[0]-60 : face_position[2]+60, :]
                scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
                return scaled