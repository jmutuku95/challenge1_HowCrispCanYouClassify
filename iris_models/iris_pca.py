'''Reducing the dimensionality of he data using PCA and the using another model
to learn from the data'''
from pickle import dump
import time
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from pandas import read_csv
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input

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
    with open(f'label_encoder.pkl', 'wb') as file:
        dump(l_encoder, file)
    return data, encoded_labels, one_hot_labels

def create_sequential_model(input_dim, output):
    '''create a sequential model, inputs is the input_dim,
    output is the dim of outputs'''
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(output, activation='relu'))
    # sgd = SGD(lr=0.1, momentum=0.9)
    model.compile(
        loss='mean_squared_error',
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model


def train_log_regression_model(x_train, y_train):
    log_regressor = LogisticRegression()
    log_regressor.fit(x_train, y_train)
    mean_score = cross_val_score(log_regressor, x_train, y_train, cv=10, scoring='accuracy').mean()
    with open(f'log_regressor_{mean_score}.pkl', 'wb') as file:
        dump(log_regressor, file)
    print(f'Mean Score was: {mean_score}.')


def reduce_dimensionality(data, components=2):
    pca = PCA(n_components=components)
    pca.fit(data)
    new_data = pca.transform(data)

    print(new_data.shape)
    # Let's see how much information we've managed to preserve
    print("-- Preserved Variance --")
    print(pca.explained_variance_ratio_)
    sum_var_ratio = sum(pca.explained_variance_ratio_)
    print(f'Total preserved variance {sum_var_ratio}')
    with open(f'pca_{sum_var_ratio}', 'wb') as file:
        dump(pca, file)
    return new_data, pca


def main():
    one_hot_labels, encoded_labels, x_train = prep_data(
        './data/iris_train.csv')
    print('LogisticRegression without PCA')
    train_log_regression_model(x_train=x_train, y_train=encoded_labels)

    print('Neural Network without PCA')
    model = create_sequential_model(x_train.shape[1], one_hot_labels.shape[1])
    model.fit(x_train, one_hot_labels, epochs=20, batch_size=3)
    model.save('ffnn_no_pca.pkl.h5')

    # Reduce dimensionality to 2 with PCA
    x_train, pca = reduce_dimensionality(x_train)

    print('LogisticRegression with PCA')
    train_log_regression_model(x_train=x_train, y_train=encoded_labels)
    print('Neural Network with PCA')
    model = create_sequential_model(x_train.shape[1], one_hot_labels.shape[1])
    model.fit(x_train, one_hot_labels, epochs=20, batch_size=3)
    model.save('ffnn_pca.pkl.h5')


if __name__ == '__main__':
    main()
