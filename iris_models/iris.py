'''A keras feedforward classifier for the iris dataset'''
from pickle import dump
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.optimizers import SGD


def create_sequential(input_dim, output):
    '''create a sequential model, inputs is the input_dim,
    output is the dim of outputs'''
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(output, activation='relu'))
    sgd = SGD(lr=0.1, momentum=0.9)
    model.compile(
        loss='mean_squared_error',
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model

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
INPUT_DIM = X_TRAIN.shape[1]
OUTPUT_DIM = Y_TRAIN.shape[1]

MODEL = create_sequential(input_dim=INPUT_DIM, output=OUTPUT_DIM)

MODEL.fit(X_TRAIN, Y_TRAIN, epochs=100, batch_size=3)
import time
MODEL.save(f'iris_clf_{time.time()}.h5')
