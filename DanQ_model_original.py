import numpy as np
import logging
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Dropout,Flatten,Dense,LSTM,Bidirectional,BatchNormalization,Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import Sequence

# logging callback to track training and validation metrics
class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LoggingCallback, self).__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.train_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))
        logging.info(f"Epoch {epoch + 1}, Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

# DataGenerator with optional normalization and augmentation
class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size, shuffle=True, normalize=False, augment_rc=False):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle, self.normalize, self.augment_rc = shuffle, normalize, augment_rc
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X, batch_y = self.X[batch_indices].astype(np.float32), self.y[batch_indices].astype(np.float32)

        if self.normalize:
            mean = np.mean(batch_X, axis=(0, 1))
            std = np.std(batch_X, axis=(0, 1))
            batch_X = (batch_X - mean) / (std + 1e-8)

        if self.augment_rc:
            for i in range(len(batch_X)):
                if np.random.rand() < 0.5:
                    batch_X[i] = batch_X[i][::-1, ::-1]
            # 添加随机噪声
            noise = np.random.normal(0, 0.01, batch_X.shape)
            batch_X = np.clip(batch_X + noise, 0, 1)

        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_model(lr):
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    with strategy.scope():
        forward_lstm = LSTM(units=320, return_sequences=True)
        backward_lstm = LSTM(units=320, return_sequences=True, go_backwards=True)
        brnn = Bidirectional(forward_lstm, backward_layer=backward_lstm)

        logging.info('building model')

        model = Sequential()
        model.add(Conv1D(filters=320,
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
        model.add(Dense(units=925,activation='relu',kernel_regularizer=l2(0.0005)))
        model.add(Dense(units=919,activation='sigmoid'))


        logging.info('compiling model')
        optimizer = RMSprop(learning_rate=lr)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer)
    return model
