import os
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from DanQ_model_original import  DataGenerator
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

custom_objects = {
    'precision_metric': precision_metric,
    'recall_metric': recall_metric,
    'dice_score': dice_score,
    'loss':'binary_crossentropy'
}

testmat = h5py.File('data/osa_data/test.mat', 'r')
print('test data loaded')
test_x = np.array(testmat['testxdata'])
test_y = np.array(testmat['testdata'])

labels = []
with open('original_data/osa/tag_osa_new.txt','r') as file:
    for line in file:
        labels.append(line.strip())
print('test labels loaded')

model = tf.keras.models.load_model('DanQ_bestmodel_osa.hdf5', custom_objects=custom_objects)
print('model loaded')

# Prediction
y_pred = model.predict(test_x)
print('model predict')

# transfrom y_pred to binary form
y_pred_binary = (y_pred > 0.5).astype(int)

# calculate roc and auc
fpr, tpr, _ = roc_curve(test_y.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

# calculate pr_auc and average precision
precision, recall, _ = precision_recall_curve(test_y.ravel(), y_pred.ravel())
average_precision = average_precision_score(test_y, y_pred, average="micro")

# ROC plot
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Micro-average)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# PRAUC plot
plt.figure()
plt.plot(recall, precision, label=f'Precision-recall curve (area = {average_precision:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Micro-average)')
plt.legend(loc="lower left")
plt.savefig('prauc_curve.png')
plt.close()

# Calculate Pearson correlation coefficient and Spearman correlation coefficient
average_precision = precision_score(test_y, y_pred_binary, average='micro', zero_division=0)
average_recall = recall_score(test_y, y_pred_binary, average='micro', zero_division=0)
average_f1 = f1_score(test_y, y_pred_binary, average='micro', zero_division=0)
pearson_corr, _ = pearsonr(test_y.ravel(), y_pred.ravel())
spearman_corr, _ = spearmanr(test_y.ravel(), y_pred.ravel())
print(f"Average precision: {average_precision:.4f}")
print(f"Average recall: {average_recall:.4f}")
print(f"Average F1: {average_f1:.4f}")
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
print(f"Spearman correlation coefficient: {spearman_corr:.4f}")

