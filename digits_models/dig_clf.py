import time
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout
from keras.utils import to_categorical

df_train = pd.read_csv('./data/digits_train.csv')
df_test = pd.read_csv('./data/digits_test.csv')
print(df_train[:5])
print(df_train.columns)
y_train = df_train['labels']
print(np.unique(y_train))
y_train_categorical = to_categorical(np.array(y_train))
print(y_train_categorical.shape)

del df_train['labels']
del df_train['Unnamed: 0']
del df_test['Unnamed: 0']
x_train = df_train
print(x_train.shape)
print(y_train.shape)
print(df_test.shape)

model = Sequential()

model.add(Dense(32,activation='relu',input_dim=64))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train_categorical, batch_size=32, epochs=100, verbose=1)

model.save('digits.h5')
