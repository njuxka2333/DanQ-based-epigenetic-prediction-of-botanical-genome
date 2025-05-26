import numpy as np
import h5py
import pandas as pd
import scipy.io
from Bio import SeqIO
from process_data import extract_label,generate_data
import tensorflow as tf
from DanQ_model import precision_metric,recall_metric,dice_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, r2_score, f1_score
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt

# define custom metric objects
custom_objects = {
    'precision_metric': precision_metric,
    'recall_metric': recall_metric,
    'dice_score': dice_score,
}

label = []
with open('original_data/osa/tag_osa_new.txt', 'r') as file:
    for line in file:
        label.append(line.strip())
print("label file loaded")

all_records = list(SeqIO.parse('original_data/osa/mergedtag_osa_1024_500.fa', 'fasta'))
all_chrom = ['1','2','3','4','5','6','7','8''9','10','11','12']
test_dataset = {chrom:[[],[]] for chrom in all_chrom}

for record in all_records:
    chrom = record.id.split('::')[1].split(':')[0]
    test_dataset[chrom][0].append(generate_data(str(record.seq)))
    test_dataset[chrom][1].append(extract_label(record.id, label))

# 转换为NumPy数组
for chrom in all_chrom:
    test_dataset[chrom][0] = np.array(test_dataset[chrom][0])
    test_dataset[chrom][1] = np.array(test_dataset[chrom][1])

print("sequence file loaded")

baseline_model = tf.keras.models.load_model('DanQ_bestmodel_osa.hdf5', custom_objects=custom_objects)
print("baseline_model loaded")
ablation_model= tf.keras.models.load_model('DanQ_bestmodel_osa_CNN.hdf5', custom_objects=custom_objects)
print("ablation_model loaded")

df = []
for chrom in all_chrom:
    
    test_x,test_y = test_dataset[chrom]
    
    #calculate test loss R square and f1 score
    result_1 = baseline_model.evaluate(test_x, test_y)
    test_loss_1,test_auc1 = result_1[0],result_1[1]
 
    y_pred = baseline_model.predict(test_x)
    y_pred_binary = np.round(y_pred)
    r2_1 = r2_score(test_y, y_pred)
    f1_1 = f1_score(test_y, y_pred_binary, average='micro')
    print(f"{chrom} data predicted by baseline_model")

    #calculate test loss R square and f1 score
    result_2 = ablation_model.evaluate(test_x, test_y)
    test_loss_2,test_auc2 = result_2[0],result_2[1]

    y_pred = ablation_model.predict(test_x)
    y_pred_binary = np.round(y_pred)
    
    r2_2 = r2_score(test_y, y_pred)
    f1_2 = f1_score(test_y, y_pred_binary, average='micro')
    print(f"{chrom} data predicted by ablation_model")

    # record test results of chrom 2 and chrom1
    df.append({
                "chromosome": chrom,
                "test loss_1": test_loss_1,
                "test loss_2": test_loss_2,
                "r2_square_1": r2_1 ,
                "r2_square_2": r2_2,
                "f1_score_1": f1_1,
                "f1_score_2": f1_2
    })

df = pd.DataFrame(df).set_index("chromosome")



fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3])  # 上0.4–1，下0–0.4
axes_upper = [plt.subplot(gs[0, i]) for i in range(3)]
axes_lower = [plt.subplot(gs[1, i]) for i in range(3)]

column_pairs = [
    ("test loss_1", "test loss_2"),
    ("r2_square_1", "r2_square_2"),
    ("f1_score_1", "f1_score_2")
]

y_labels = ["test loss", "R square", "F1 score"]

for i, (col1, col2) in enumerate(column_pairs):
    for ax in [axes_upper[i], axes_lower[i]]:
        sns.boxplot(data=df[[col1, col2]], color=None, showcaps=True,
                    boxprops=dict(facecolor="none", edgecolor="k"),
                    whiskerprops=dict(color="k"),
                    medianprops=dict(color="k"), ax=ax)
        sns.stripplot(data=df[[col1, col2]], color='black', alpha=0.5, ax=ax)

    axes_upper[i].set_ylim(0.4, 1)  # 上半部分显示 0.4 到 1
    axes_lower[i].set_ylim(0, 0.4)  # 下半部分显示 0 到 0.4

    for ax in [axes_upper[i], axes_lower[i]]:
        ax.set_xticklabels(["Baseline model", "Ablation model"])
        ax.set_ylabel(y_labels[i])

    # 去掉上轴的x轴和下轴的上边框
    axes_upper[i].spines['bottom'].set_visible(False)
    axes_lower[i].spines['top'].set_visible(False)
    axes_upper[i].tick_params(labelbottom=False)
    axes_lower[i].xaxis.tick_bottom()
    axes_upper[i].xaxis.tick_top()

    # 添加断裂的斜线
    d = .015  # 斜线尺寸
    kwargs = dict(transform=axes_upper[i].transAxes, color='k', clip_on=False)
    axes_upper[i].plot((-d, +d), (-d, +d), **kwargs)
    axes_upper[i].plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=axes_lower[i].transAxes)
    axes_lower[i].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axes_lower[i].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

plt.suptitle('Baseline vs. Ablation Model Comparison')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Ablation_experiment_osa_broken_axis.png')

df.to_csv('Ablation_experiment_osa.csv')
