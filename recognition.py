import cv2
import sys
import os
import facenet
import tensorflow as tf
import numpy as np
import detect_face
import pickle
import matplotlib.pyplot as plt

def realtime_recognition(image):
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    image_size = 160

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            facenet.load_model('models')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser('new_models.pkl')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.ndim == 2:
                gray = facenet.to_rgb(gray)

                bounding_boxes, points = detect_face.detect_face(gray, minsize, pnet, rnet, onet ,threshold, factor)
                nrof_faces = bounding_boxes.shape[0]

            for face_position in bounding_boxes:
                face_position = face_position.astype(int)

                cropped = gray[face_position[1]:face_position[3],face_position[0]:face_position[2],:]

                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    continue

                scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
                plt.imshow(scaled)
                scaled = scaled.reshape(-1, image_size, image_size, 3)

                emb_array = sess.run(embeddings, feed_dict={images_placeholder: scaled, phase_train_placeholder: False})

                # predictions = model.predict_proba(emb_array)
                predict = model.predict(emb_array)
                # print(predict)
                
                cv2.rectangle(image, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (255, 255, 0), 2)
                cv2.putText(image, class_names[predict[0]] , (face_position[0], face_position[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), thickness=2, lineType=2)

                if points.shape[0] != 0:
                    for i in range(points.shape[1]):
                        count = points.shape[0]/2
                        count = int(count)
                        for j in range(count):
                            cv2.circle(image, (points[j][i], points[j + count][i]), 3, (255, 255, 0), -1)

    return image, predict