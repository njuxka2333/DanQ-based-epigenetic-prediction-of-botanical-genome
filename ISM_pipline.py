import torch
import numpy as np
import os
import numpy as np
import torch
import os
import sys

from utils.data_utils import *
from utils.data import NucDataset  # 自定义 PyTorch Dataset
from models.model_architectures import sei_model,SeiX_model
from utils.preprocessing import *
from utils.graph_utils import plot_auroc_auprc
from utils.preprocess import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import psutil, os
from logomaker import Logo
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

custom_colors = {
    'A': '#CC0000',
    'T': '#008000',
    'C': '#0000CC',
    'G': '#FFB300'
}

def save_seq_to_fasta(seq, chrom, start, region, save_path):
    """
    将指定区间的序列保存为标准FASTA格式

    参数：
        seq: 字符串序列（ATCG），长度为1024（或更长）
        chrom: 染色体号（如 'A07'）
        start: 原始序列对应的基因组起始位置（即 seq[0] 在基因组上的位置）
        region: (region_start, region_end)，相对于基因组坐标（即保存哪个区域）
        save_path: 输出FASTA路径，例如 "./output.fa"
    """
    region_start, region_end = region
    offset_start = start + region_start
    offset_end = start + region_end

    assert 0 <= region_start < region_end <= len(seq), "region 超出序列边界"

    region_seq = seq[region_start:region_end]
    header = f">{chrom}:{offset_start}-{offset_end}"

    with open(save_path, 'w') as f:
        f.write(header + '\n')
        f.write(region_seq + '\n')

    print(f"✅ FASTA 已保存至: {save_path}，长度: {len(region_seq)}bp")

def load_model_tags(tag_file_path):
    with open(tag_file_path) as f:
        tags = [line.strip() for line in f if line.strip()]
    return tags

def get_feature_column_index(target_name, tag_file):
    tags = load_model_tags(tag_file)
    if target_name not in tags:
        raise ValueError(f"特征 '{target_name}' 不在模型输出标签中！可选值包括: {tags}")
    return tags.index(target_name)

def make_tag_dict(tag_file):
    # 加载标签文件，并创建标签字典
    with open(tag_file) as f:
        tag_list = f.readlines()

    tag_dict = {item.strip(): index for index, item in enumerate(tag_list)}
    return tag_dict

def single_nucl_mutation(seq):
    """
    对序列中的每个碱基进行单碱基突变，返回突变后的独热编码。
    包括原始序列，共 1 + 1024*3 = 3073 个序列。
    """
    db = ['A', 'C', 'G', 'T']
    seqlist = [seq]  # 原始序列在最前

    for i, nuc in enumerate(seq):
        for alt in db:
            if alt != nuc:
                alt_seq = seq[:i] + alt + seq[i+1:]
                seqlist.append(alt_seq)

    alt_pre = NucPreprocess(seqlist)
    X_alt = alt_pre.onehot_for_nuc()

    return X_alt


from Bio import SeqIO


def extract_seqs_from_fa_by_csv(fa_path, csv_path):
    """
    根据 csv 文件中定义的多个区域，从 fa 文件中批量提取长度为 1024 的碱基序列。

    参数：
        fa_path: fasta 文件路径（包含多条染色体）
        csv_path: CSV 文件路径，需包含 'chrom', 'start', 'end' 三列

    返回：
        List[str]: 序列列表（与 csv 顺序一致）
    """
    expected_len = 1024
    print(f"📖 加载参考基因组: {fa_path}")

    # 预读取所有染色体序列
    chrom_seqs = {record.id: str(record.seq).upper() for record in SeqIO.parse(fa_path, "fasta")}
    print(f"✅ 载入 {len(chrom_seqs)} 条染色体序列")

    # 读取csv
    df = pd.read_csv(csv_path)
    sequences = []

    for i, row in df.iterrows():
        chrom, start, end = str(row["chrom"]), int(row["start"]), int(row["end"])

        if chrom not in chrom_seqs:
            raise ValueError(f"[❌ 错误] 在 {fa_path} 中未找到染色体: {chrom}")

        sub_seq = chrom_seqs[chrom][start:end]
        if len(sub_seq) != expected_len:
            raise ValueError(f"[❌ 错误] 第 {i} 行 ({chrom}:{start}-{end}) 提取到序列长度为 {len(sub_seq)}，不是 1024")

        sequences.append(sub_seq)

    print(f"✅ 成功提取 {len(sequences)} 条序列（每条 1024bp）")
    return sequences

def predict_ref_profiles_batch(
    seqs,                   # List[str]，ATCG序列，每条长度为1024
    model_path,             # 模型路径
    model_tag_file,         # 模型 tag 文件（决定输出特征数）
    batch_size=512,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
):
    """
    对传入的 N 条序列进行非突变预测，返回每条序列的所有染色质特征预测值。

    返回：
        numpy.ndarray，shape = (N, num_features)
    """
    # Step 1: One-hot 编码
    preprocessor = NucPreprocess(seqs)
    X_ref = preprocessor.onehot_for_nuc()  # shape: (N, 1024, 4)

    # Step 2: 构建 DataLoader
    ref_dataset = NucDataset(x=X_ref, y=None)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Step 3: 加载模型
    tag_dict = make_tag_dict(model_tag_file)
    num_features = len(tag_dict)
    model = sei_model.Sei(sequence_length=1024, n_genomic_features=num_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Step 4: 预测
    predictions = []
    with torch.no_grad():
        for batch_inputs in ref_loader:
            batch_inputs = batch_inputs.to(device, dtype=torch.float)
            batch_inputs = batch_inputs.permute(0, 2, 1)  # (B, 4, 1024)
            batch_outputs = model(batch_inputs)
            predictions.append(batch_outputs.cpu().numpy())

    return np.concatenate(predictions, axis=0)  # shape: (N, num_features)

def predict_mutation_delta_by_split(seqs, model_path, model_tag_file, batch_size=512, device="cpu"):
    """
    支持多个序列的 ISM 批量预测：ref 和 mutation 分别预测

    返回：
        List[np.ndarray]: 每条序列的 Δ 分数数组 (3072, n_feature)
    """
    from torch.utils.data import DataLoader
    model_tag_dict = make_tag_dict(model_tag_file)
    n_feature = len(model_tag_dict)

    # Step 1: 拆分所有 ref 和 mutation 编码
    X_refs = []
    X_muts = []

    for seq in seqs:
        assert len(seq) == 1024, f"序列长度必须为1024，当前为{len(seq)}"
        X_alt = single_nucl_mutation(seq)  # 包含ref + 3072突变
        X_refs.append(X_alt[0])           # 取第0行是ref
        X_muts.append(X_alt[1:])            # 取第1~3072行为突变

    # 将每个小list中的 tensor 提取出来拼接成一个大 array
    # 转换为 NumPy 后拼接
    X_refs_all = np.stack([x.cpu().numpy() for x in X_refs], axis=0)  # shape: (N, 1024, 4)

    # 扁平化，把 X_muts 中的所有突变扁平展开
    flat_mut_list = []

    for mut_list in X_muts:
        for mut in mut_list:
            # mut 应该是 (1024, 4) 的 Tensor 或 list
            if isinstance(mut, list):
                mut = torch.tensor(mut, dtype=torch.float32)
            flat_mut_list.append(mut)
    X_muts_all = torch.stack(flat_mut_list, dim=0).cpu().numpy()  # shape = (N*3072, 1024, 4)

    # Step 2: 模型加载
    model = sei_model.Sei(sequence_length=1024, n_genomic_features=n_feature)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    def predict(X_all):
        dataset = NucDataset(x=X_all, y=None)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device, dtype=torch.float).permute(0, 2, 1)
                outputs = model(inputs).squeeze()
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(0)
                all_preds.append(outputs.cpu().numpy())
        return np.concatenate(all_preds, axis=0)  # shape: (N, n_feature)

    # Step 3: 执行预测
    preds_ref = predict(X_refs_all)                # shape: (N, n_feature)
    preds_mut = predict(X_muts_all)                # shape: (N*3072, n_feature)

    # Step 4: 计算 Δ 分数
    delta_scores = []
    for i in range(len(seqs)):
        ref_i = preds_ref[i]                       # shape: (n_feature,)
        mut_i = preds_mut[i*3072:(i+1)*3072]       # shape: (3072, n_feature)
        delta = mut_i - ref_i                      # shape: (3072, n_feature)
        delta_scores.append(delta)

    return delta_scores  # List of (3072, n_feature)

def run_and_plot_boxplot(seqs_peak, seqs_nonpeak, model_path, model_tag_path, histone_name,
                         batch_func, result_prefix, result_path, title_suffix="", mutation=False, batch_size=512, device=None):
    """
    简洁封装：对peak与non-peak区域序列进行预测、统计、绘图

    参数说明：
        seqs_peak: list[str]，Peak区域ATCG序列
        seqs_nonpeak: list[str]，Non-Peak区域ATCG序列
        model_path: 模型路径
        model_tag_path: 标签路径
        histone_name: 目标组蛋白名（如 H3K4ME3）
        batch_func: 预测函数（可以是 predict_ref_profiles_batch 或 predict_mutation_delta_by_split）
        result_prefix: 输出图像前缀
        title_suffix: 图标题补充说明（如 "原始预测"、"ISM突变"）
        mutation: 是否为ISM突变，影响 slicing 逻辑
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import mannwhitneyu

    def pval_to_star(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return 'n.s.'

    tag_index = load_model_tags(model_tag_path).index(histone_name)

    # 预测
    pred_peak = batch_func(seqs_peak, model_path, model_tag_path, batch_size, device)
    pred_nonpeak = batch_func(seqs_nonpeak, model_path, model_tag_path, batch_size, device)

    if mutation:
        # 先拼接 list 中的所有 ndarray
        peak_arr = np.concatenate(pred_peak, axis=0)  # shape = (N×3072, F)
        nonpeak_arr = np.concatenate(pred_nonpeak, axis=0)

        # 然后提取对应特征并取绝对值
        vals_peak = np.abs(peak_arr[:, tag_index])
        vals_nonpeak = np.abs(nonpeak_arr[:, tag_index])
    else:
        vals_peak = np.abs(pred_peak[:, tag_index].flatten())
        vals_nonpeak = np.abs(pred_nonpeak[:, tag_index].flatten())

    # 整合 DataFrame
    df = pd.DataFrame({
        "Δscore (abs)": np.concatenate([vals_peak, vals_nonpeak]),
        "Region": ["Peak"] * len(vals_peak) + ["Non-Peak"] * len(vals_nonpeak)
    })

    # 显著性检验
    stat, p_value = mannwhitneyu(vals_peak, vals_nonpeak, alternative="greater")

    # 绘图
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df, x="Region", y="Δscore (abs)", width=0.5, palette="Set2", showfliers=False)
    plt.title(f"{histone_name} {title_suffix}", fontsize=14)
    plt.ylabel("")
    plt.xlabel("")
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    save_path = f"{result_prefix}_{'mutation' if mutation else 'itself'}.png"
    plt.savefig(os.path.join(result_path, save_path), dpi=300)
    plt.close()
    print(f"✅ 图已保存至: {save_path}")


def predict_with_mutation_scan(seqs, model_path, model_tag_file, device, batch_size=512):
    """
    接收一个 1024bp 的序列，执行突变扫描并返回 3073×n_feature 的预测矩阵

    参数：
        seq (str): 1024 长度的 ATCG 序列
        model_path (str): 训练好的模型路径
        device (torch.device): 'cuda' or 'cpu'
        n_feature (int): 模型输出特征维度
        batch_size (int): 预测批大小

    返回：
        total_pre (np.ndarray): shape = (3073, n_feature)
    """

    # Step 1: 单碱基突变并编码
    X_alt = single_nucl_mutation(seq)
    print(f"突变编码序列总数: {len(X_alt)}")

    # Step 2: 构建 DataLoader
    pre_dataset = NucDataset(x=X_alt,  y=None)
    pre_loader = DataLoader(dataset=pre_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_tag_dict = make_tag_dict(model_tag_file)
    model = sei_model.Sei(sequence_length=1024, n_genomic_features=len(model_tag_dict))  # 仍然使用完整的 tag_dict

    # Step 3: 加载模型
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Step 4: 执行预测
    predictions = []
    with torch.no_grad():
        for inputs in pre_loader:
            inputs = inputs.to(device, dtype=torch.float)
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    return predictions

def plot_single_feature_mutation_delta(pred_scores, start=0, figsize=(30, 5)):
    """
    可视化单个染色质特征在 1024 个碱基上突变带来的 Δscore

    参数：
        pred_scores: shape=(3073,) 或 (3073, 1) 的 array，原始 + 突变序列的预测值
        start: 起始坐标（用于横轴标注）
        figsize: 图像尺寸
    """

    pred_scores = np.ravel(pred_scores)  # 确保是一维向量
    assert len(pred_scores) == 3073, "预测值数量必须为 3073（1原始 + 1024×3突变）"

    ref_score = pred_scores[0]
    mut_scores = pred_scores[1:]  # shape: (3072,)
    delta_scores = mut_scores - ref_score

    # 构造绘图数据
    df_plot = pd.DataFrame({
        "delta": delta_scores,
        "alt": ["alt" + str(i % 3 + 1) for i in range(3072)],
        "pos": [i // 3 + 1 for i in range(3072)]
    })

    # 可视化
    plt.figure(figsize=figsize)
    sns.lineplot(data=df_plot, x="pos", y="delta", hue="alt", palette="tab10")
    plt.axhline(0, color='gray', linestyle='--')
    plt.xticks(range(0, 1024, 128), range(start, start + 1024, 128))
    plt.xlabel("Genomic Position")
    plt.ylabel("Δ Score (Mutated - Original)")
    plt.title("Single Nucleotide Mutation Impact on Feature")
    plt.tight_layout()
    plt.show()

def build_mutation_matrix(delta_scores, ref_seq, species_name, histone_name, result_path=".", start=0, region=(399, 601), figsize=(30, 6)):
    """
    构建并绘制突变敏感性热图矩阵（每个位点三种突变 + 参考碱基）

    参数：
        delta_scores: shape=(3072,) 的 array，表示每个突变后的 Δ 分数（mut - ref）
        ref_seq: 原始 1024bp 的 ATCG 序列（长度必须为 1024）
        start: 起始坐标（用于坐标轴）
        region: 热图绘制区间，如 (399, 601)
        figsize: 图像尺寸
        save_path: 若提供路径则保存图像

    返回：
        values_result: shape=(1024, 4) 的突变热图矩阵（0 表示参考碱基）
    """
    assert len(delta_scores) == 3072, "必须为 1024×3 个突变 Δscore"
    assert len(ref_seq) == 1024, "参考序列长度必须为 1024"

    # Step 1：构造 dataframe 表格
    df = pd.DataFrame({
        "delta": delta_scores,
        "pos": np.repeat(np.arange(1, 1025), 3),  # 1~1024 每个位点 3 个突变
        "alt": ["alt" + str(i % 3 + 1) for i in range(3072)]
    })

    # Step 2：将每个位点的突变结果转为矩阵，并补充参考碱基为 0
    ref_db = ['A', 'C', 'G', 'T']
    values_result = []

    for i in range(1024):
        # 找出不是参考碱基的三个碱基（模拟 alt）
        ref_base = ref_seq[i].upper()
        alt_bases = [b for b in ref_db if b != ref_base]
        delta_vals = df.iloc[i * 3:i * 3 + 3]["delta"].values

        # 将突变分数插入参考碱基位置（参考碱基为 0）
        ref_idx = ref_db.index(ref_base)
        values_result.append(np.insert(delta_vals, ref_idx, 0))

    values_result = np.array(values_result)  # shape = (1024, 4)

    # Step 3：绘图
    plt.figure(figsize=figsize)
    region_start, region_end = region
    sns.heatmap(
        values_result[region_start:region_end, :].T,
        cmap='vlag',
        center=0,
        xticklabels=range(region_start + start, region_end + start),
        yticklabels=ref_db
    )

    # 自动控制刻度间隔
    region_length = region_end - region_start
    max_ticks = 20  # 最多显示 20 个刻度
    tick_step = max(1, region_length // max_ticks)

    xtick_pos = range(0, region_length, tick_step)
    xtick_labels = [region_start + start + i for i in xtick_pos]
    plt.xticks(ticks=xtick_pos, labels=xtick_labels, rotation=0)


    plt.xlabel("Genomic Position")
    plt.ylabel("Nucleotide")
    plt.title("Single Nucleotide Mutation Sensitivity Matrix")
    plt.tight_layout()
    save_path = os.path.join(result_path, f"{species_name}_{histone_name}_{region_start+ start}_{region_end + start}_mutation_matrix.png")
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"热图已保存至: {save_path}")
    else:
        plt.show()

    return values_result

def plot_reference_base_importance(values_result, ref_seq, start=0, region=(399, 601), figsize=(30, 6), species_name="", histone_name="", result_path="."):
    """
    绘制每个位置参考碱基的重要性热图

    参数：
        values_result: shape=(1024, 4)，突变影响矩阵（含参考碱基为 0）
        ref_seq: 原始序列（长度为 1024）
        start: 基因组起始坐标
        region: 可视化范围 (默认 399~600)
        save_path: 是否保存图像
    """
    ref_db = ['A', 'C', 'G', 'T']
    region_start, region_end = region

    # 每个位置只有一个碱基重要性值，其余为 0
    logo_result = []
    for i in range(1024):
        ref_base = ref_seq[i].upper()
        ref_idx = ref_db.index(ref_base)
        alt_vals = np.delete(values_result[i], ref_idx)  # 取非参考碱基的突变值
        abs_sum = np.sum(np.abs(alt_vals))

        temp = [0] * 4
        temp[ref_idx] = abs_sum
        logo_result.append(temp)

    logo_result = np.array(logo_result)

    # 画图
    plt.figure(figsize=figsize)
    sns.heatmap(
        logo_result[region_start:region_end, :].T,
        cmap="vlag",
        center=0,
        xticklabels=range(region_start + start, region_end + start),
        yticklabels=ref_db,
        cbar_kws={"label": "Reference Base Importance"}
    )

    # 自动稀疏 X 轴坐标
    region_length = region_end - region_start
    max_ticks = 20
    tick_step = max(1, region_length // max_ticks)
    xtick_pos = range(0, region_length, tick_step)
    xtick_labels = [region_start + start + i for i in xtick_pos]
    plt.xticks(ticks=xtick_pos, labels=xtick_labels, rotation=0)

    plt.title("Reference Base Importance (Δscore sum of mutations)")
    plt.xlabel("Genomic Position")
    plt.ylabel("Reference Base")
    plt.tight_layout()

    save_path = os.path.join(result_path,
                             f"{species_name}_{histone_name}_{region_start + start}_{region_end + start}_ref_base_importance.png")
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"热图已保存至: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model_path = "../../../../models/saved_models/merged_tag_HFT_BC/ath/mergedtag_ath_HFT_BC_20250312_203749_1024_nip_feature7.model"
    model_path = "../../../../models/saved_models/merged_tag_HFT_BC/cross/Score_regression_cross_osa_zma_HFT_BC_20250312_171305_1024_nip_feature7.model"
    model_tag_path = "shuffle_tag_osa_zma.txt"
    result_path = "./sorghum_bicolor"
    species_name = "sorghum_bicolor"
    histone_name = "H3K4ME3"
    region=(399, 601)
    region_start, region_end = region
    peak_csv_file = os.path.join(result_path,"SbMC_H3K4me3_Rep2.cpm_top50_peaks.csv")
    nonpeak_csv_file = os.path.join(result_path, "SbMC_H3K4me3_Rep2.cpm_top50_flanking_nonpeaks.csv")
    # 读取 ATCG 序列 peak峰值区域
    peak_seqs = extract_seqs_from_fa_by_csv(fa_path=f"D:\\data\\tag_file\\fa\\{species_name}\\{species_name}.fa",
                                      csv_path=peak_csv_file)
    nonpeak_seqs = extract_seqs_from_fa_by_csv(fa_path=f"D:\\data\\tag_file\\fa\\{species_name}\\{species_name}.fa",
                                               csv_path=nonpeak_csv_file)
    # # 非突变预测
    run_and_plot_boxplot(
        seqs_peak=peak_seqs,
        seqs_nonpeak=nonpeak_seqs,
        model_path=model_path,
        model_tag_path=model_tag_path,
        histone_name=histone_name,
        batch_func=predict_ref_profiles_batch,
        result_prefix=f"{histone_name}_compare",
        result_path=result_path,
        title_suffix="Prediction Score",
        mutation=False,
        device=device
    )

    # # 突变预测
    run_and_plot_boxplot(
        seqs_peak=peak_seqs,
        seqs_nonpeak=nonpeak_seqs,
        model_path=model_path,
        model_tag_path=model_tag_path,
        histone_name=histone_name,
        batch_func=predict_mutation_delta_by_split,
        result_prefix=f"{histone_name}_compare",
        result_path=result_path,
        title_suffix="In Situ Mutation Effect",
        mutation=True,
        device=device
    )

    # results_peak_itself = predict_ref_profiles_batch(
    #     seqs=peak_seqs,
    #     model_path=model_path,
    #     model_tag_file=model_tag_path,
    #     batch_size=512
    # )
    # results_peak_Mutation = predict_mutation_delta_by_split(
    #     seqs=seqs,
    #     model_path=model_path,
    #     model_tag_file=model_tag_path,
    #     batch_size=512,
    #     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # )
    # # 读取 ATCG 序列 nonpeak区域
    # nonpeak_seqs = extract_seqs_from_fa_by_csv(fa_path=f"D:\\data\\tag_file\\fa\\{species_name}\\{species_name}.fa",
    #                                    csv_path=nonpeak_csv_file)
    # results_nonpeak_itself = predict_ref_profiles_batch(
    #     seqs=nonpeak_seqs,
    #     model_path=model_path,
    #     model_tag_file=model_tag_path,
    #     batch_size=512
    # )
    # results_nonpeak_Mutation = predict_mutation_delta_by_split(
    #     seqs=seqs,
    #     model_path=model_path,
    #     model_tag_file=model_tag_path,
    #     batch_size=512,
    #     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # )

    # # 保存
    # np.save("results_peak.npy", results_peak)
    # np.save("results_nonpeak.npy", results_nonpeak)

    # === 1. 加载数据 ===
    # results_peak = np.load("results_peak.npy")
    # results_nonpeak = np.load("results_nonpeak.npy")

    # # === 2. 选择你关注的染色质特征 ===
    # target_index = load_model_tags(model_tag_path).index(histone_name)
    #
    # # 取出该特征的 Δscore 值（已经是 mut - ref）
    # delta_peak = results_peak_itself[:, target_index]
    # delta_nonpeak = results_nonpeak_itself[:, target_index]
    #
    # # === 3. 绝对值：表示变化幅度 ===
    # abs_peak = np.abs(delta_peak).flatten()
    # abs_nonpeak = np.abs(delta_nonpeak).flatten()
    #
    # # === 4. 整合为 DataFrame ===
    # df = pd.DataFrame({
    #     "Δscore (abs)": np.concatenate([abs_peak, abs_nonpeak]),
    #     "Region": ["Peak"] * len(abs_peak) + ["Non-Peak"] * len(abs_nonpeak)
    # })
    #
    # # ==== 统计显著性检验 ====
    # stat, p_value = mannwhitneyu(abs_peak, abs_nonpeak, alternative="greater")
    # print(f"📊 Mann-Whitney U 检验: p = {p_value:.4e}")
    #
    # # ==== 绘图 ====
    # plt.figure(figsize=(6, 6))
    # sns.boxplot(data=df, x="Region", y="Δscore (abs)", width=0.5, palette="Set2", showfliers=False)
    # # sns.stripplot(data=df, x="Region", y="Δscore (abs)", color=".3", size=1.5, alpha=0.15)
    #

