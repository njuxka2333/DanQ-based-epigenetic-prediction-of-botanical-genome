# DanQ_model
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        return batch_X, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def create_model(lr):
    with strategy.scope():
        forward_lstm = LSTM(units=128, return_sequences=True)
        backward_lstm = LSTM(units=128, return_sequences=True, go_backwards=True)
        brnn = Bidirectional(forward_lstm, backward_layer=backward_lstm)

        logging.info('building model')
        print('building model')

        model = Sequential()
        model.add(Conv1D(filters=128,
                         kernel_size=26,
                         input_shape=(1024, 4),
                         padding="valid",
                         activation="relu",
                         strides=1,
                         groups=1))

        model.add(MaxPooling1D(pool_size=13, strides=13))

        model.add(Dropout(0.2))

        model.add(brnn)

        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(units=256))
        model.add(Activation('relu'))

        model.add(Dense(units=29))
        model.add(Activation('sigmoid'))

        logging.info('compiling model')
        print('compiling model')
        optimizer = RMSprop(learning_rate=lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model
