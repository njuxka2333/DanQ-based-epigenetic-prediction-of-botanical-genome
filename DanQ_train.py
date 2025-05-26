import os
import h5py
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 修改导入模块名称，假设优化后的模型文件名为 danq_model_optimized.py
from DanQ_model_pureCNN import DataGenerator, create_model, LoggingCallback, precision_metric
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.stats import pearsonr, spearmanr

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_DISABLE'] = '1'
np.random.seed(1337)  # for reproducibility
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logging.info('loading data')

trainmat = h5py.File('data/osa_data/train.mat', 'r')
logging.info('train.mat loaded')
validmat = h5py.File('data/osa_data/valid.mat', 'r')
logging.info('valid.mat loaded')


X_train, y_train = np.array(trainmat['trainxdata']), np.array(trainmat['traindata'])
valid_x, valid_y = np.array(validmat['validxdata']), np.array(validmat['validdata'])


logging.info('building model')
model = create_model(lr=8.674559776324777e-05  # lr
batch_size = 80 # batch size

# Data Generator
train_generator = DataGenerator(X_train, y_train, batch_size=batch_size, shuffle=True, normalize=True, augment_rc=True)
val_generator = DataGenerator(valid_x, valid_y, batch_size=batch_size, shuffle=True, normalize=True, augment_rc=True)

# checkpointer
checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel_zma_CNN.hdf5", verbose=1, save_best_only=True)

# earlystopper
earlystopper = EarlyStopping(monitor='val_dice_score', patience=3, verbose=1)

# learning rate scheduler
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)

# logging callback
logging_callback = LoggingCallback()

# training model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # epoch number
    callbacks=[checkpointer, logging_callback],
    workers=1,
    use_multiprocessing=False
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')

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

plt.savefig('training_validation_metrics.png')
plt.close()
