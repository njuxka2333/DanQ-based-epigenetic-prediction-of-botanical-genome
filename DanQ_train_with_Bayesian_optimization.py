import os
import h5py
import logging
import scipy.io
import numpy as np
import tensorflow as tf
from DanQ_model import DataGenerator, create_model, LoggingCallback, precision_metric
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skopt import gp_minimize
from skopt.space import Integer, Real
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_DISABLE'] = '1'
np.random.seed(1337)  # for reproducibility
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

logging.basicConfig(level=logging.INFO)
logging.info('loading data')

trainmat = h5py.File('data/osa_data/train.mat', 'r')
logging.info('train.mat loaded')
validmat = h5py.File('data/osa_data/valid.mat', 'r')
logging.info('valid.mat loaded')

X_train, y_train = np.array(trainmat['trainxdata']), np.array(trainmat['traindata'])
valid_x, valid_y = np.array(validmat['validxdata']), np.array(validmat['validdata'])

space = [
    Integer(80, 112, name='batch_size'),
    Integer(20, 35, name='epochs'),
    Real(1e-6, 1e-3, prior='log-uniform', name='lr')
]


def objective(params):
    batch_size, epochs, lr = params
    print("currment hyperparameters: batch_size={}, epochs={}, lr={}".format(batch_size, epochs, lr))

    model = create_model(lr)

    train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
    valid_generator = DataGenerator(valid_x, valid_y, batch_size=batch_size)

    checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
    logging_callback = LoggingCallback()

    history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator,
                        callbacks=[checkpointer, logging_callback], workers=1, use_multiprocessing=False)
    max_dice_score = max(history.history['val_dice_score'])
    return -max_dice_score


# 使用 forest_minimize 进行超参数优化
result = gp_minimize(objective, space, n_calls=5,n_initial_points=3,   random_state=0)
best_params = result.x
best_val_dice_score = -result.fun
print("最佳超参数组合: batch_size={}, epochs={}, lr={}".format(best_params[0], best_params[1], best_params[2]))

final_model = create_model( best_params[2])

train_generator = DataGenerator(X_train, y_train, batch_size=best_params[0])
valid_generator = DataGenerator(valid_x, valid_y, batch_size=best_params[0])

checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel_zma.hdf5", verbose=1, save_best_only=True)
logging_callback = LoggingCallback()

history = final_model.fit(train_generator, epochs=best_params[1], validation_data=valid_generator,
                          callbacks=[checkpointer, logging_callback], workers=1, use_multiprocessing=False)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve_osa.png')

fig, axes = plt.subplots(2, 4, figsize=(18, 12))
# training loss with batch
axes[0, 0].plot(logging_callback.train_losses, label='Training Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Batch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
# training precision with batch
axes[0, 1].plot(logging_callback.train_precisions, label='Training Precision')
axes[0, 1].set_title('Training Precision')
axes[0, 1].set_xlabel('Batch')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].legend()
# training recall with batch
axes[0, 2].plot(logging_callback.train_recalls, label='Training Recall')
axes[0, 2].set_title('Training Recall')
axes[0, 2].set_xlabel('Batch')
axes[0, 2].set_ylabel('Recall')
axes[0, 2].set_ylim(0, 1)
axes[0, 2].legend()
# training dice_score with batch
axes[0, 3].plot(logging_callback.train_dice_scores, label='Training Dice Score')
axes[0, 3].set_title('Training Dice Score')
axes[0, 3].set_xlabel('Batch')
axes[0, 3].set_ylabel('Dice Score')
axes[0, 3].set_ylim(0, 1)
axes[0, 3].legend()
# valid loss
axes[1, 0].plot(logging_callback.val_losses, label='Validation Loss')
axes[1, 0].set_title('Validation Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
# valid precision
axes[1, 1].plot(logging_callback.val_precisions, label='Validation Precision')
axes[1, 1].set_title('Validation Precision')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()
# valid recall
axes[1, 2].plot(logging_callback.val_recalls, label='Validation Recall')
axes[1, 2].set_title('Validation Recall')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Recall')
axes[1, 2].set_ylim(0, 1)
axes[1, 2].legend()
# valid dice_score
axes[1, 3].plot(logging_callback.val_dice_scores, label='Validation Dice Score')
axes[1, 3].set_title('Validation Dice Score')
axes[1, 3].set_xlabel('Epoch')
axes[1, 3].set_ylabel('Dice Score')
axes[1, 3].set_ylim(0, 1)
axes[1, 3].legend()

plt.savefig('training_validation_metrics_osa.png')
plt.close()
