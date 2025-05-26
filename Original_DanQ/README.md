# DanQ 模型使用说明

## 1. DanQ 模型简介

DanQ 是一个结合了卷积神经网络（CNN）与双向长短期记忆网络（BLSTM）的深度学习模型，最初由 Quang 和 Xie 于 2016 年提出，用于预测 DNA 序列的功能性区域，特别是在转录因子结合位点和表观遗传标记等基因调控任务中展现出强大性能。

### 模型结构如下：

- **输入层**  
  原始 DNA 序列通常以 one-hot 编码形式输入，例如长度为 1,000 bp 的序列转化为 1000×4 的矩阵。

- **卷积层（CNN）**  
  提取局部序列特征，相当于模拟生物中 motif 的识别功能。使用多个卷积核，每个核可以学习不同的 motif 模式。

- **池化层（MaxPooling）**  
  降维并保留局部最强特征，通常用于压缩卷积结果，提高计算效率。

- **双向 LSTM 层（BLSTM）**  
  捕捉序列中的长距离依赖信息。相较于传统 CNN 模型（如 DeepSEA），加入 Bi-LSTM 后，模型可以理解序列片段之间的上下文关系。

- **全连接层（Dense）与输出层**  
  最终通过 sigmoid 激活函数输出每个标签的概率，支持多标签（二元）分类。

---

## 2. DanQ 使用流程

### Step 1️⃣：准备输入数据

- **序列提取**：从 ChIPHub 数据中提取定长序列（1024 bp），并以染色体编号为单位划分训练/验证/测试集。
- **one-hot 编码**：将 DNA 序列转为形状为 `(L, 4)` 的 one-hot 向量（A, C, G, T 分别对应 `[1,0,0,0]` 等）。
- **标签二值分类**：将标签信息转化为 `1×n`（n 为标签数）的向量，1 表示存在该标签，0 表示不存在。

---

### Step 2️⃣：构建 DanQ 模型结构

```python
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Flatten, Dense
from tensorflow.keras.models import Model

# 简化版结构示意
input = Input(shape=(1000, 4))
x = Conv1D(filters=320, kernel_size=26, activation='relu')(input)
x = MaxPooling1D(pool_size=13, strides=13)(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(320, return_sequences=True))(x)
x = Flatten()(x)
x = Dense(925, activation='relu')(x)
output = Dense(num_labels, activation='sigmoid')(x)
model = Model(inputs=input, outputs=output)
```

### Step 3️⃣：训练模型

- 使用 `model.fit(...)` 或自定义 `DataGenerator` 实现分批训练；
- 常配合 `EarlyStopping`、`ModelCheckpoint` 等回调函数以控制训练；
- 数据集较大时，建议使用 `tf.data.Dataset` 或多线程的数据生成器以提高效率。

---

### Step 4️⃣：模型评估与可视化

- 评估指标包括：
  - **AUC**（ROC曲线下面积）
  - **AUPRC**（精确率-召回率曲线下面积）
  - **F1-score**
  - **Pearson 相关系数**
  - **Spearman 相关系数**
- 可视化方式：
  - 绘制 ROC 和 PR 曲线；
  - 结合 In Silico Mutagenesis（ISM）生成突变热图；
  - 叠加 BigWig 文件实现模型预测与实验数据对比。

---

## 3. 使用方法

### 3.1 获取 DanQ 代码

```bash
git clone https://github.com/njuxka2333/DanQ-based-epigenetic-prediction-of-botanical-genome.git
cd DanQ-based-epigenetic-prediction-of-botanical-genome/Original_DanQ
```
该仓库包含基于 TensorFlow 重构的 DanQ 模型代码，适配 Python 3.x 环境。原始版本依赖 Python 2.7，已不再兼容现代环境。
