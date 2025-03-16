import numpy as np
import h5py
import scipy.io
import os
import argparse
import logging
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.utils import Sequence


logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument("--train_filepath",help="train file")
parser.add_argument("--valid_filepath",help="valid file")
parser.add_argument("--test_filepath",help="test file")
parser.add_argument("--output_model_filepath",help="output model")
parser.add_argument("--output_loss_curve_filepath",help="output loss curve")

args = parser.parse_args()

# 设置只使用单 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置 NCCL 环境变量
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_DISABLE'] = '1'

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


np.random.seed(1337)  # for reproducibility

# 配置单 GPU 策略
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

logging.basicConfig(level=logging.INFO)
logging.info('loading data')
print('loading data')

# 检查文件是否存在
train_file_path = args.train_filepath
valid_file_path = args.valid_filepath
test_file_path =  args.test_filepath


trainmat = h5py.File(train_file_path)
logging.info("使用 h5py 成功读取训练文件。")
print("使用 h5py 成功读取训练文件。")

validmat = h5py.File(valid_file_path, 'r')
logging.info("使用 h5py 成功读取验证文件。")
print("使用 h5py 成功读取验证文件。")

testmat = h5py.File(test_file_path, 'r')
logging.info("使用 h5py 成功读取测试文件。")
print("使用 h5py 成功读取测试文件。")


# 后续代码保持不变
X_train = np.array(trainmat['trainxdata'])
y_train = np.array(trainmat['traindata'])
valid_x = np.array(validmat['validxdata'])
valid_y = np.array(validmat['validdata'])
test_x = np.array(testmat['testxdata'])
test_y = np.array(testmat['testdata'])


def create_model(lr):
    with strategy.scope():
        forward_lstm = LSTM(units=320, return_sequences=True)
        backward_lstm = LSTM(units=320, return_sequences=True, go_backwards=True)
        brnn = Bidirectional(forward_lstm, backward_layer=backward_lstm)

        logging.info('building model')
        print('building model')

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

        model.add(Dense(units=925))
        model.add(Activation('relu'))

        model.add(Dense(units=58))
        model.add(Activation('sigmoid'))

        logging.info('compiling model')
        print('compiling model')
        optimizer = RMSprop(learning_rate=lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# 定义超参数搜索空间
batch_sizes = [32, 48, 64, 80, 96]
epochs_list = [15, 30, 45, 60]
learning_rates = [0.01, 0.0001, 0.00001, 0.000001]

best_val_loss = float('inf')
best_params = None

# 遍历所有超参数组合
for batch_size, epochs, lr in product(batch_sizes, epochs_list, learning_rates):
    print(f"当前超参数组合: batch_size={batch_size}, epochs={epochs}, lr={lr}")

    model = create_model(lr)

    checkpointer = ModelCheckpoint(filepath=args.output_model_filepath, verbose=1, save_best_only=True)

    train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
    valid_generator = DataGenerator(valid_x, valid_y, batch_size=batch_size)

    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                logging.info(f'Epoch {epoch + 1}: loss = {logs.get("loss")}, val_loss = {logs.get("val_loss")}')

    logging_callback = LoggingCallback()

    history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator,
                        callbacks=[checkpointer, logging_callback], workers=1, use_multiprocessing=False)

    val_loss = min(history.history['val_loss'])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = (batch_size, epochs, lr)

print(f"最佳超参数组合: batch_size={best_params[0]}, epochs={best_params[1]}, lr={best_params[2]}")

# 使用最优超参数训练最终模型
best_batch_size, best_epochs, best_lr = best_params

final_model = create_model(best_lr)

checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel_final.hdf5", verbose=1, save_best_only=True)

train_generator = DataGenerator(X_train, y_train, batch_size=best_batch_size)
valid_generator = DataGenerator(valid_x, valid_y, batch_size=best_batch_size)

logging_callback = LoggingCallback()

history = final_model.fit(train_generator, epochs=best_epochs, validation_data=valid_generator,
                          callbacks=[checkpointer, logging_callback], workers=1, use_multiprocessing=False)

tresults = final_model.evaluate(test_x, test_y)
logging.info(f"Test results: {tresults}")

# 绘制训练损失和验证损失曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 保存图像
plt.savefig(args.output_loss_curve_filepath)

