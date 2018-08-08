'''Iris classification model using knns'''
from pickle import dump
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from pandas import read_csv
import numpy as np
from keras.utils import to_categorical

def prep_data(path):
    '''read data from a csv file and binarize the  labels'''
    data = read_csv(path)
    data = shuffle(data)
    labels = data['labels']
    unique_labels = np.unique(labels)
    n_classes = unique_labels.shape[0]

    l_encoder = LabelEncoder()
    del data['labels']

    l_encoder.fit(labels)
    print(f'Encoder classes: { l_encoder.classes_ }')

    encoded_labels = l_encoder.transform(labels)
    print(f'Example Encoded labels:\n { encoded_labels[:7] }')

    one_hot_labels = to_categorical(encoded_labels, num_classes=n_classes)
    print(f'One hot labels:\n { one_hot_labels[:7] }')
    return l_encoder, one_hot_labels, data


IRIS_PATH = './data/iris_train.csv'
IRIS_TEST = read_csv('./data/iris_test.csv')
LABEL_ENCODER, Y_TRAIN, X_TRAIN = prep_data(IRIS_PATH)


KNN = KNeighborsClassifier(n_neighbors=20)


print(cross_val_score(KNN, X_TRAIN, Y_TRAIN, cv=10, scoring='accuracy').mean())
with open(f'knn_{time.time()}', 'wb') as file:
    dump(KNN, file)
