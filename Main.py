import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv
import time

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def classification():

    #ucitavanje i pretpocesiranje
    start = time.time()
    train_dir = 'train/'
    test_dir = 'test/'
    train_dict = {}
    test_dict = {}

    with open('train/train_labels.csv', mode='r') as infile:
        reader = csv.reader(infile)

        headers = next(reader)
        for rows in reader:
            train_dict[rows[0]] = rows[1]

    with open('test/test_labels.csv', mode='r') as infile:
        # data = pandas.read_csv('train/train_labels.csv', encoding='latin-1', engine='python')
        reader = csv.reader(infile)

        headers = next(reader)
        for rows in reader:
            test_dict[rows[0]] = rows[1]

    train_images = {}
    test_images = {}

    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        if img_name == "train_labels.csv":
            break;
        img = load_image(img_path)
        resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        train_images[img_name] = resized

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        if img_name == "test_labels.csv":
            break;
        img = load_image(img_path)
        resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        test_images[img_name] = resized

    #hog feature extraction

    img_features = []
    test_img_features = []

    train_labels = []
    test_labels = []

    cell_size = (40, 40)
    block_size = (3,3)

    hog = cv2.HOGDescriptor(_winSize=(resized.shape[1] // cell_size[1] * cell_size[1],
                                      resized.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=9)

    for img in train_images:
        img_features.append(hog.compute(train_images[img]))
        train_labels.append(train_dict[img])

    for img in test_images:
        test_img_features.append(hog.compute(test_images[img]))
        test_labels.append(test_dict[img])

    x_train = np.array(img_features)
    nsamples, nx, ny = x_train.shape
    x_train = x_train.reshape((nsamples, nx * ny))
    y_train = np.array(train_labels)

    x_test= np.array(test_img_features)
    nsamples, nx, ny = x_test.shape
    x_test = x_test.reshape((nsamples, nx * ny))
    y_test = np.array(test_labels)

    #svm obucavanje

    svm = SVC(kernel="linear", decision_function_shape='ovo')
    svm.fit(x_train, y_train)
    y_train_pred = svm.predict(x_train)
    y_test_pred = svm.predict(x_test)
    x = 0
    for i in range(len(y_test)):
        #print(str(i) + ") " + y_test[i] + " : " + y_test_pred[i])
        if y_test[i] == y_test_pred[i]:
            x += 1
    #print(x/80)
    print("Train set accuracy: ", accuracy_score(y_train, y_train_pred)*100, '%')
    print("Test set accuracy: ", round(accuracy_score(y_test, y_test_pred)*100, 2), '%')
    end = time.time()

classification()
