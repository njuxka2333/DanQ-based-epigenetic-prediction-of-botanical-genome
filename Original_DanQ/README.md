DanQ模型使用说明
1. DanQ模型简介
DanQ 是一个结合了卷积神经网络（CNN）与双向长短期记忆网络（BLSTM）的深度学习模型，最初由 Quang 和 Xie 于 2016 年提出，用于预测 DNA 序列的功能性区域，特别是在转录因子结合位点和表观遗传标记等基因调控任务中展现出强大性能。
￮	模型结构如下：
▪	输入层
•	原始 DNA 序列通常以 one-hot 编码形式输入，例如长度为 1,000 bp 的序列转化为 1000×4 的矩阵。
▪	卷积层（CNN）
•	提取局部序列特征，相当于模拟生物中“motif”的识别功能；
•	使用多个卷积核，每个核可以学习不同的motif模式。
▪	池化层（MaxPooling）
•	降维并保留局部最强特征，通常用于压缩卷积结果，提高计算效率。
▪	双向LSTM层（BLSTM）
•	捕捉序列中的长距离依赖信息；
•	相较于传统的CNN模型（如DeepSEA），加入Bi-LSTM后，模型可以理解序列片段之间的上下文关系。
▪	全连接层（Dense）与输出层
•	最终通过 sigmoid 激活函数输出每个标签的概率，支持多标签（二元）分类。
 
2. DanQ使用流程
Step 1️⃣：准备输入数据
•	序列提取：从chiphub数据中中提取定长序列（1024bp），并以染色体编号为单位划分训练/验证/测试集。
•	one-hot 编码：将 DNA 序列转为形状为 (L, 4) 的 one-hot 向量（A, C, G, T 分别对应 [1,0,0,0] 等）。
•	标签二值分类：将标签信息转化成1xn（n为标签数）的向量，1代表序列存在该标签类别，0代表不存在
Step2️⃣：构建DanQ模型结构
Python
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
•	注意：num_labels 根据数据集而定（如 DeepSEA 是 919；植物可自定义为 Histone 标签个数）。
•	优化器常用 RMSprop，损失函数为 binary crossentropy。
Step 3️⃣：训练模型
•	使用 model.fit(...) 或自定义 DataGenerator 实现分批训练；
•	常配合 EarlyStopping、ModelCheckpoint 等回调；
•	数据集较大时建议使用 tf.data.Dataset 或自定义生成器支持多线程读取。
Step 4️⃣：模型评估与可视化
评估指标：AUC、AUPRC、F1-score、Pearson相关系数与Spearman相关系数等；
3. 使用方法
3.1 获取DanQ代码
Plain Text
git clone https://github.com/njuxka2333/DanQ-based-epigenetic-prediction-of-botanical-genome.git
cd DanQ-based-epigenetic-prediction-of-botanical-genome/Original_DanQ
DanQ-based-epigenetic-prediction-of-botanical-genome仓库下存储了基于tensorflow重构的DanQ模型代码，原代码仅支持python 2.7版本，与目前常用版本不兼容
3.2 配置python环境
Plain Text
conda create -n danq_env python=3.8
conda activate danq_env
pip install -r requirements.txt 
3.3 构建数据集
根据源数据的格式，从以下选一种构建DeepsEA数据集
Plain Text
python build_DeepSEA_data.py\
--tag_flie original/tag.txt #替换标签文件路径
--train_valid_file original_data/mergedtag_1024_512.fa #替换用于构建训练和验证集的fasta文件路径
--test_file original_data/mergedtag_1024_500.fa #替换用于构建测试集的fasta文件路径
-- train
--train_filename data/train.mat \ #训练文件
--valid_filename data/valid.mat \ #验证文件
--test_filename  data/test.mat \ # 测试文件
可视化数据的标签与染色体分布（选做）
Plain Text
python EDA.py
3.4 模型训练
Plain Text
python DanQ_train.py
如果你需要通过贝叶斯优化寻找最佳超参数（批次大小，训练周期数，学习率），运行以下脚本：
Plain Text
python DanQ_train_with_Bayesian_optimization.py
贝叶斯优化采用高斯过程作为先验估计函数，并结合概率上限策略在搜索空间内选择每一轮的评估点。你可以在代码中调整优化次数和设定监控指标（valid loss等）
3.5 模型评估
Plain Text
python DanQ_test.py
生成ROC曲线、PRAUC曲线
计算平均F1分数、平均精准率和平均召回率
计算y_pred和y_true之间的Pearson相关系数与Spearman相关系数


