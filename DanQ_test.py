import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
from DanQ_model import precision_metric,recall_metric,dice_score,soft_iou_loss, combined_loss
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, f1_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# load test data
testmat = h5py.File('data/test.mat', 'r')
test_x = np.array(testmat['testxdata'])
test_y = np.array(testmat['testdata'])

#load labels
labels = []
with open('original_data/osa/tag_osa.txt','r') as file:
    for line in file:
        labels.append(line.strip())

# define custom metric objects
custom_objects = {
    'precision_metric': precision_metric,
    'recall_metric': recall_metric,
    'dice_score': dice_score,
    'soft_iou_loss': soft_iou_loss,
    'combined_loss': combined_loss
}

#load trained model
model = tf.keras.models.load_model('DanQ_bestmodel_fold.hdf5', custom_objects=custom_objects)


# Prediction
y_pred = model.predict(test_x)

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
pearson_corr, _ = pearsonr(test_y.ravel(), y_pred.ravel())
spearman_corr, _ = spearmanr(test_y.ravel(), y_pred.ravel())
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
print(f"Spearman correlation coefficient: {spearman_corr:.4f}")

# Calculate the accuracy, precision, and F1 score for each category
num_classes = test_y.shape[1]
df = {}
for i in range(num_classes):
    class_accuracy = accuracy_score(test_y[:, i], y_pred_binary[:, i])
    class_precision = precision_score(test_y[:, i], y_pred_binary[:, i])
    class_f1 = f1_score(test_y[:, i], y_pred_binary[:, i])
    df[labels[i]] = {" 准确率":class_accuracy,"精确率":class_precision,"F1 分数":class_f1}
df =pd.DataFrame(df)
print(df)
