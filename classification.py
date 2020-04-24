import tensorflow as tf
import numpy as np
import cv2
import facenet
import sys
import os
from os.path import join
import pickle
import random
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# def get_Batch(data, label, batch_size):
#     # print(data.shape, label.shape)
#     input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32)
#     x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
#     # y_batch = (y_ba.eval()).tolist()
#     return x_batch, y_batch

# def get_data_Batch(inputs, batch_size, shuffle=False):
#     rows = len(inputs[0])
#     indices = list(range(rows))

#     if shuffle:
#        random.seed(100)
#        random.shuffle(indices)

#     while True:
#         batch_indices = np.asarray(indices[0:batch_size])
#         indices = indices[batch_size:] + indices[:batch_size]
#         batch_data = []
#         for data in inputs:
#             data = np.asarray(data)
#             temp_data = data[batch_indices]
#             batch_data.append(temp_data.tolist())
#         yield batch_data
def train():
    def get_next_batch(batch):
        return batch.__next__()

    def get_image_batch(image, minsize):
        rows = len(image[0])
        indices = list(range(rows))

        while True:
            batch_indices = np.asarray(indices[0:minsize])
            indices = indices[minsize:] + indices[:minsize]
            image_data = []
            for i in image:
                i = np.asarray(i)
                temp = i[batch_indices]
                image_data.append(temp.tolist())
            yield image_data

    def load_data(image_paths, image_size):
        nrof_samples = len(image_paths)
        images = []
        for i in range(nrof_samples):
            # print(image_paths[i])
            img = cv2.imread(image_paths[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.ndim == 2:
                img = facenet.to_rgb(gray)
            
            images.append(img)
        # x = np.array(images)
        # print(x.shape)
        return images

    # data_dir = './train_data'
    data_dir = '/home/wjc/Documents/code/face_test/images'
    image_size = 160
    # batch_size = 50

    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed = 42)
            dataset = facenet.get_dataset(data_dir)

            paths, labels = facenet.get_image_paths_and_labels(dataset)
            # print('Number of classes: %d' % len(dataset))
            # print('Number of images: %d' % len(paths))

            # print('Loading feature extraction model')
            facenet.load_model('/home/wjc/Documents/code/face_test/models')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]

            images = load_data(paths, image_size)
            iamge_batch = get_image_batch(image=[images, labels], minsize=150)
            # print(images[0].shape, len(images))
            epoch = int(len(labels)/150) + 1
            classifier_filename_exp = os.path.expanduser('new_models.pkl')
            
            for i in range(epoch):
                images_batch, labels_batch = get_next_batch(iamge_batch)

                emb_array = sess.run(embeddings, feed_dict = {images_placeholder:images_batch, phase_train_placeholder:False})
                # print('emb_array.shape:')
                # print(emb_array.shape)
                # print(labels)
                # print(type(labels))

                X_train, X_test, y_train, y_test = train_test_split(emb_array, labels_batch, test_size=.2, random_state=42)
                # print(type(X_train))
                # print(type(y_train))
                # X_Batch, y_Batch = get_Batch(X_train, y_train, 50)
                
                # print(X_Batch.shape)
                # print(type(y_Batch.eval()))

                # classifier_filename_exp = os.path.expanduser('new_models.pkl')

                # print('Training classifier')
                # model = SVC(kernel='poly',degree=10,gamma=1,coef0=0,probability=True)
                model = SGDClassifier(loss='log', penalty='l2')
                # model = KNeighborsClassifier()

                # coord = tf.train.Coordinator()
                # threads = tf.train.start_queue_runners(sess, coord)
                # epoch = 0
                # try:
                #     while not coord.should_stop():
                #         X_Batch, y_Batch = get_Batch(X_train, y_train, 1)
                #         # print(type(X_Batch.eval()))
                #         # print(type(y_Batch.eval()))
                #         # y_Batch = y_Batch.eval(session = sess)
                #         # Y_Batch = y_Batch.tolist()
                #         # print(type(Y_Batch))
                #         model.partial_fit(X_Batch, y_Batch)
                #         epoch += 1
                # except tf.errors.OutOfRangeError:
                #     print('---Train end---')
                # finally:
                #     coord.request_stop()
                #     print('---Program end---')
                # coord.join(threads)

                # batch = get_data_Batch(inputs=[X_train, y_train],batch_size=50,shuffle= False)

                # for i in range(5):
                #     X_batch, y_batch = get_next_batch(batch)
                #     # print(y_batch)
                #     # print(np.unique(y_batch))
                #     model.partial_fit(X_batch, y_batch, classes = np.unique(y_train))

                model.partial_fit(X_train, y_train, classes = np.unique(labels))
                
                class_names = [cls.name.replace('_',' ') for cls in dataset]
                # print(class_names)

                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                # print('Saved classifier model to file "%s"' % classifier_filename_exp)

                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                predict = model.predict(X_test)
                accuracy = metrics.accuracy_score(y_test, predict)
                # print('accuracy: %.2f%%' % (100 * accuracy))


