import numpy as np
import h5py
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument("--test_filepath",help="test file")
parser.add_argument("--model_filepath",help="model")


args = parser.parse_args()

# 加载测试数据
test_file_path = args.test_filepath
testmat = h5py.File(test_file_path, 'r')
test_x = np.array(testmat['testxdata'])
test_y = np.array(testmat['testdata'])

# 加载训练好的模型
model = tf.keras.models.load_model(args.model_filepath)

# 进行预测
y_pred = model.predict(test_x)

# 计算微平均的 ROC 曲线和 AUC
fpr, tpr, _ = roc_curve(test_y.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

# 计算微平均的 PRAUC 曲线和平均精度
precision, recall, _ = precision_recall_curve(test_y.ravel(), y_pred.ravel())
average_precision = average_precision_score(test_y, y_pred, average="micro")

# 绘制 ROC 曲线
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

# 绘制 PRAUC 曲线
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

# 计算皮尔森相关系数和斯皮尔曼相关系数
pearson_corr, _ = pearsonr(test_y.ravel(), y_pred.ravel())
spearman_corr, _ = spearmanr(test_y.ravel(), y_pred.ravel())
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
print(f"Spearman correlation coefficient: {spearman_corr:.4f}")
