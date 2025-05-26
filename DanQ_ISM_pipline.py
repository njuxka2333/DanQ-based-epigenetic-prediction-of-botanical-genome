import os
import logging
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
import pyBigWig
import tensorflow as tf
from tensorflow.keras.models import load_model
from DanQ_model import precision_metric, recall_metric, dice_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import mannwhitneyu,pearsonr
import logomaker
from sklearn.metrics import roc_auc_score, average_precision_score

sns.set_theme(style='white', font_scale=1.2)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)

def select_test_data(original_fa_path,ISM_fa_path,all_chroms,frac = 0.01,seed =42):
    logging.info(f" reference genome loaded: {original_fa_path}")
    random.seed(seed)
    sample_seqs = []
    all_seqs_divid_chrom ={chrom:[] for chrom in all_chroms}
    for record in SeqIO.parse(original_fa_path, "fasta"):
        chrom =record.id.split("::")[1].split(":")[0]
        all_seqs_divid_chrom[chrom].append(record)
    
    for chrom in all_chroms:
        sample_seqs += random.sample(all_seqs_divid_chrom[chrom],int(frac*len(all_seqs_divid_chrom[chrom])))
    
    with open(ISM_fa_path,'w') as f:
        for record in sample_seqs:
            f.write('>'+str(record.id) + '\n')
            f.write(str(record.seq)+'\n')
        
def extract_seqs_from_fa(fa_path):
    logging.info(f" ISM data loaded: {fa_path}")
    chroms = []
    starts = []
    ends = []
    seqs = []
    for record in SeqIO.parse(fa_path, "fasta"):
        chrom = record.id.split('::')[1].split(':')[0]
        start,end = record.id.split('::')[1].split(':')[1].split('-')
        start = int(start)
        end = int(end)
        seq = str(record.seq).upper()
        chroms.append(chrom)
        starts.append(start)
        ends.append(end)
        seqs.append(seq)
    logging.info(f"  {len(seqs)} sequences loaded")
    return chroms,starts,ends,seqs

def load_tags(tag_path):
    with open(tag_path) as f:
        tags = [line.strip() for line in f if line.strip()]
        logging.info("tags loaded")
    return tags

def get_tag_dict(tags):
    return {tag: idx for idx, tag in enumerate(tags)}

def load_chip_flie(bw_path):
    bw = pyBigWig.open(bw_path)
    logging.info(f"{bw_path} loaded")
    return bw

def extract_chip_signal(bw, chrom, start, end):
    signal = bw.values(chrom, start, end, numpy=True)
    return np.nan_to_num(signal)
    
def load_trained_model(model_path):
    custom_objects = {
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'dice_score': dice_score,
        'loss':'binary_crossentropy'
    }
    model = load_model(model_path, custom_objects=custom_objects)
    print('model load')
    
    return model

def one_hot_encode(seqs):
    n = len(seqs)
    encoded = np.zeros((n, 1024, 4), dtype=np.float32)
    db = ['A', 'G', 'C', 'T']
    
    for i, seq in enumerate(seqs):
        for j, char in enumerate(seq):
            if char in db:
                encoded[i, j, db.index(char)] = 1.0
    return encoded

def single_nucl_mutation(seq):
    db = ['A', 'C', 'G', 'T']
    seqlist = [seq]
    
    # 生成所有单碱基突变
    for i, nuc in enumerate(seq):
        for alt in db:
            if alt != nuc:
                alt_seq = seq[:i] + alt + seq[i+1:]
                seqlist.append(alt_seq)
    return one_hot_encode(seqlist)

def predict(model: tf.keras.Model, seqs, batch_size= 64):
    return model.predict(seqs, batch_size=batch_size)

def run_ism(model, tag_dict, seq, feature_name):
    if feature_name not in tag_dict:
        raise ValueError(f" '{feature_name}'is not in the tag list of DanQ model")
    
    feature_idx = tag_dict[feature_name]
    db = ['A', 'C', 'G', 'T']
    
    encoded_seqs= single_nucl_mutation(seq)
    
    predictions = predict(model, encoded_seqs)
    
    ref_value = predictions[0, feature_idx]
    mut_values = predictions[1:, feature_idx]
    
    delta_scores = mut_values - ref_value
    
    delta_matrix = np.zeros((1024, 4))
    
    for i in range(1024):
        for j, base in enumerate(db):
            ref_base = seq[i]
            if base == ref_base:
                delta_matrix[i, j] = 0
            else:
                mut_idx = i * 3 + (j if j < db.index(ref_base) else j - 1)
                delta_matrix[i, j] = delta_scores[mut_idx]
    return ref_value, delta_matrix
    
def run_multi_ism(model, seq):
    db = ['A', 'C', 'G', 'T']
    
    encoded_seqs= single_nucl_mutation(seq)
    
    predictions = predict(model, encoded_seqs)
    
    ref_value = predictions[0,:]
    mut_values = predictions[1:,:]  #

    delta_scores = mut_values - ref_value
    
    return delta_scores, ref_value
    
def build_delta_matrix(delta_scores,seq):
    db = ['A', 'C', 'G', 'T']
    delta_matrix = np.zeros((1024, 4))
    
    for i in range(1024):
        for j, base in enumerate(db):
            ref_base = seq[i]
            if base == ref_base:
                delta_matrix[i, j] = 0
            else:
                mut_idx = i * 3 + (j if j < db.index(ref_base) else j - 1)
                delta_matrix[i, j] = delta_scores[mut_idx]
    delta_matrix_norm = (delta_matrix - np.mean(delta_matrix))/np.std(delta_matrix)
    return delta_matrix_norm
    
def analyze_region_importance(delta_matrix, seq):
    db = ['A', 'C', 'G', 'T']
    position_importance = np.zeros(1024)
    
    for i in range(1024):
        ref_base_idx = db.index(seq[i])
        alt_vals = np.delete(delta_matrix[i], ref_base_idx)
        position_importance[i] = np.mean(np.abs(alt_vals))

    return position_importance

def compute_pearson_correl(position_importance ,chip_signal):
    corr, _ = pearsonr(position_importance, chip_signal)
    return corr

def plot_mutation_heatmap(delta_matrix,start = 0 ,region = None, title ="Reference  delta Score",save_path=None):
    ref_db  = ['A', 'C', 'G', 'T']
    
    if region is None:
        region = (0, 1024)
    
    region_start, region_end = region
    
    plt.figure(figsize=(30,2))
    ax = sns.heatmap(
        delta_matrix[region_start:region_end, :].T,
        cmap= "coolwarm",
        center=0,
        xticklabels=range(region_start + start, region_end + start),
        yticklabels=ref_db,
        cbar_kws={'label': 'delta Score'},
    )
    
    plt.xticks(ticks=[], labels=[], rotation=0)
    plt.title("")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_sequence_logo(delta_matrix, sequence,start = 0 ,region = None,title ="Reference Base Importance",save_path=None):
    db= ['A', 'C', 'G', 'T']
    nucleotide_colors = {
        'A': '#CC0000',  # 红色
        'G': '#FFB300',  # 橙色
        'C': '#0000CC',  # 蓝色
        'T': '#008000',  # 绿色
    }
    
    position_importance = analyze_region_importance(delta_matrix,sequence)
    if region is None:
        region = (0, 1024)
    
    region_start, region_end = region
    region_sequence = sequence[region_start:region_end]
    region_importance = position_importance[region_start:region_end]
    
    logo_data = np.zeros((region_end-region_start, 4))
    for i, base in enumerate(region_sequence):
        base_idx = db.index(base)
        logo_data[i, base_idx] = region_importance[i]
    
    df = pd.DataFrame(logo_data, columns=db)
    
    logo = logomaker.Logo(
        df,
        color_scheme=nucleotide_colors,
        figsize=(120, 2)
    )
    
    plt.xticks(ticks=[], labels=[], rotation=0)
    plt.title("")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
def plot_combined_visualization(delta_matrix, sequence,position_importance ,chip_signal, region=None, save_path=None):
    ref_db = ['A', 'C', 'G', 'T']
    nucleotide_colors = {
        'A': '#CC0000',  # 红色
        'G': '#FFB300',  # 橙色
        'C': '#0000CC',  # 蓝色
        'T': '#008000',  # 绿色
    }

    if region is None:
        region = (0, 1024)

    region_start, region_end = region
    region_sequence = sequence[region_start:region_end]

    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(60, 4), gridspec_kw={'height_ratios': [1, 1,1]})

    region_importance = position_importance[region_start:region_end]

    logo_data = np.zeros((region_end - region_start, 4))
    for i, base in enumerate(region_sequence):
        base_idx = ref_db.index(base)
        logo_data[i, base_idx] = region_importance[i]

    df = pd.DataFrame(logo_data, columns=ref_db)

    logo = logomaker.Logo(
        df,
        color_scheme=nucleotide_colors,
        ax=ax1
    )

    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_title("")

    ax2.plot(chip_signal, color='lightgreen', linewidth=1.5)
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_title("")


    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="0.25%", pad=0.02)

    sns.heatmap(
        delta_matrix[region_start:region_end, :].T,
        cmap="coolwarm",
        center=0,
        yticklabels=ref_db,
        cbar_ax=cax,
        cbar_kws={'label': 'delta Score'},
        ax=ax3
    )

    ax3.set_xticks([])
    ax3.set_xticklabels([])

    ax3.set_title("")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
   
if __name__ == "__main__":
    model_path = "DanQ_bestmodel_osa.hdf5"
    tag_path = "original_data/osa/tag_osa_new.txt"
    target_tag_path = "original_data/osa/tag_osa_ISM.txt"
    original_fa_path = "original_data/osa/mergedtag_osa_1024_500.fa"
    ISM_fa_path ="original_data/osa/ISM_osa.fa"
    chip_seq_path = "chip_seq/osa"
    output_root = "ISM_result"
    
    all_chroms = ['1','2','3','4','5','6','7','8','9','10','11','12']
    select_test_data(original_fa_path,ISM_fa_path,all_chroms)

    model = load_trained_model(model_path)
    
    tag_dict = get_tag_dict(load_tags(tag_path))
    target_tags = load_tags(target_tag_path)
    
    chroms,starts,ends,seqs = extract_seqs_from_fa(ISM_fa_path)
    delta_scores,ref_values = [run_multi_ism(model,seq) for seq in seqs]


            
    os.makedirs(output_root, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    i = 0
    df = {tag:[] for tag in target_tags }
    for tag in target_tags :
        logging.info(f" Analyzing label:{tag}")
        tag_idx = tag_dict[tag]

        tag_output_dir = os.path.join(output_root, tag)
        os.makedirs(tag_output_dir, exist_ok=True)
        
        bw = pyBigWig.open(os.path.join(chip_seq_path,f"{tag}.bw"))
        
        correlations = []
        for (i,seq,delta_scores,ref_values,chrom,start,end) in enumerate(zip(seqs, delta_scores,ref_values,chroms, starts, ends)):
            logging.info("analyzing {i+1} sequence")
            ref_value = ref_values[tag_idx]
            delta_score = delta_scores[:,tag_idx].T
            delta_matrix = build_delta_matrix(delta_score,seq)
            position_importance = analyze_region_importance(delta_matrix,seq)
            
            chip_signal = extract_chip_signal(bw, chrom, start, end)

            positive_combined_image_num = 1
            if ref_value >= 0.95 and positive_combined_image_num <= 10:
                plot_combined_visualization(delta_matrix, seq,position_importance ,chip_signal,  region= None, save_path=os.path.join(tag_output_dir,f"{tag}_{chrom}_{start}_{end}_figure.png"))
                logging.info(f"{tag}_{chrom}_{start}_{end}_figure created")

            negative_combined_image_num = 1
            if ref_value <= 0.05 and negative_combined_image_num <=3:
                plot_combined_visualization(delta_matrix, seq,position_importance ,chip_signal,  region= None, save_path=os.path.join(tag_output_dir,f"{tag}_{chrom}_{start}_{end}_figure.png"))
                logging.info(f"{tag}_{chrom}_{start}_{end}_figure created")

            corr = compute_pearson_correl(position_importance ,chip_signal)
            correlations.append(corr)
        
        df[tag] = correlations
        
        sns.violinplot(data=correlations,ax = axes[i], inner="box", color="skyblue", cut=0)
        axes[i].set_xlabel(f"{tag[4:]}" )
        
        if i == 0:
            axes[i].set_ylabel("Correlation")

    plt.suptitle("ISM Delta vs ChIP-seq Signal Correlation", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(ISM_fa_path,"osa_ISM_violiin_plot"))
    
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(ISM_fa_path,"ISM_osa_corr.csv"))

