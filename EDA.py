"""Build the DeepSEA dataset."""
import random
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Generate a binary table between labels and chromosomes
def EDA_stat(record_path,label_path):
    records = list(SeqIO.parse(record_path, 'fasta'))
    chroms = list({record.id.split('::')[1].split(':')[0] for record in  records})
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            labels.append(line.strip())

    df = {}
    for label in labels:
        df[label] = {chrom: 0 for chrom in chroms}

    chrom_num = dict.fromkeys(chroms, 0)

    for record in tqdm(records, desc="Processing training and valid data"):
        chrom = record.id.split('::')[1].split(':')[0]
        chrom_num[chrom] += 1
        label_info = record.id
        for label in labels:
            if label in label_info:
                df[label][chrom] += 1

    df = pd.DataFrame(df)
    df['count'] = chrom_num
    return df
 
if __name__ == "__main__":
    osa =  EDA_stat(record_path = 'orignial_data/osa/mergedtag_osa_1024_512.fa',label_path ='orignial_data/osa/tag_osa.txt')
    ath =  EDA_stat(record_path = 'orignial_data/ath/mergedtag_ath_1024_512.fa',label_path ='orignial_data/ath/tag_ath.txt')
    zma =  EDA_stat(record_path = 'orignial_data/zma/mergedtag_zma_1024_512.fa',label_path ='orignial_data/zma/tag_osa.txt')
    
    # save results in ead.xlsx
    with pd.ExcelWriter('eda.xlsx', engine='openpyxl') as writer:
        osa.to_excel(writer, sheet_name='osa')
        ath.to_excel(writer, sheet_name='ath')
        zma.to_excel(writer, sheet_name='zma')


