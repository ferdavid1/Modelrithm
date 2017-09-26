'''
Reference used for sklearn-image recognition functions:
http://www.ippatsuman.com/2014/08/13/day-and-night-an-image-classifier-with-scikit-learn/

'''

import os
from PIL import Image
from io import BytesIO
import io
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import sys
from Modelrithm import Modelrithm

def process_image(image, blocks=4):

    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]

def process_image_file(image_path):

    image_fp = BytesIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None

def process_directory(directory):

    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path)
            if img_feature:
                training.append(img_feature)
    return training

def train(training_path_a, training_path_b):

    if not os.path.isdir(training_path_a):
        raise IOError('%s is not a directory' % training_path_a)
    if not os.path.isdir(training_path_b):
        raise IOError('%s is not a directory' % training_path_b)
    training_a = process_directory(training_path_a)
    training_b = process_directory(training_path_b)
    # data contains all the training data (a list of feature vectors)
    data = training_a + training_b
    # '1' for class A and '0' for class B
    target = [1] * len(training_a) + [0] * len(training_b)
    # split training data in a train set and a test set. The test set will
    # contain 20% of the total
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20)

    # from sklearn.tree import DecisionTreeClassifier
    clf = SVC().fit(x_train, y_train)
    print('Training classifier...\n')
    print("Accuracy: {}".format(accuracy_score(y_test, clf.predict(x_test))))
    print("Precision: {}".format(precision_score(y_test, clf.predict(x_test))))
    print("F-beta Score: {}".format(fbeta_score(y_test, clf.predict(x_test), beta=0.1)))
    print(Modelrithm.Classification(x_train, x_test, y_train, y_test))

    return clf


train('images/earth', 'images/mars')