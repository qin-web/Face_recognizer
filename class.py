
import tensorflow as tf
import numpy as np
import cv2
import facenet
import os
from os.path import join 
import sys
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
        
def load_data(image_paths, image_size):
    nrof_samples = len(image_paths)
    images = []
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])
        #print(image_paths[i])
        #print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.ndim == 2:
            img = facenet.to_rgb(gray)
        images.append(img)
    return images
 
data_dir = '/home/wjc/Documents/code/face_test/train_data'
image_size = 160
 
with tf.Graph().as_default():
      
    with tf.Session() as sess:
            
        np.random.seed(seed = 42)
        dataset = facenet.get_dataset(data_dir)
        
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
           
        # 加载模型,模型位于models目录下
        print('Loading feature extraction model')
        facenet.load_model('models')
            
        # 获取输入和输出 tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        
        images = load_data(paths, image_size)
        #plt.imshow(images[10])
        
        feed_dict = {images_placeholder:images, phase_train_placeholder:False }
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        print('emb_array.shape:')
        print(emb_array.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(emb_array, labels, test_size=.3, random_state=42)
                      
        classifier_filename_exp = os.path.expanduser('new_models.pkl')
 
        # Train classifier
        print('Training classifier')
        #model = KNeighborsClassifier() # accuracy: 77.70%
        #model = SVC(kernel='linear', probability=True)
        #model = SVC(kernel='poly',degree=2,gamma=1,coef0=0,probability=True) # accuracy: 77.03%
        model = SVC(kernel='poly',degree=10,gamma=1,coef0=0,probability=True) #accuracy: 87.16%
        
        model.fit(X_train, y_train)
            
        # Create a list of class names
        class_names = [ cls.name.replace('_', ' ') for cls in dataset]
        print(class_names)
        
        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
        
        # 验证
        # with open(classifier_filename_exp, 'rb') as infile:
        #     (model, class_names) = pickle.load(infile)
        # predict = model.predict(X_test) 
        # accuracy = metrics.accuracy_score(y_test, predict)  
        # print ('accuracy: %.2f%%' % (100 * accuracy)  )
