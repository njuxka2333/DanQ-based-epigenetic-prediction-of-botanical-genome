import os
import pyBigWig
import numpy as np

def average_bigwig(rep1_path, rep2_path, output_path):
    bw1 = pyBigWig.open(rep1_path)
    bw2 = pyBigWig.open(rep2_path)

    chroms = bw1.chroms()
    
    out_bw = pyBigWig.open(output_path, "w")
    out_bw.addHeader(list(chroms.items()))

    for chrom in chroms:
        length = chroms[chrom]
        values1 = bw1.values(chrom, 0, length, numpy=True)
        values2 = bw2.values(chrom, 0, length, numpy=True)

        # Handle missing values (e.g., None or nan)
        values1 = np.nan_to_num(values1)
        values2 = np.nan_to_num(values2)

        avg_values = (values1 + values2) / 2

        out_bw.addEntries(chrom, list(range(length)), values=avg_values)

    bw1.close()
    bw2.close()
    out_bw.close()

data_path = 'chip_seq/osa'

# 使用方式
target_tags = ['osa_H3K4me3','osa_H3K27me3','H3K4me1','osa_H3K9me2','osa_H3']
for tag in target_tags:
    rep1_path = os.path.join(data_path,f"{tag}_rep1.bw")
    rep2_path = os.path.join(data_path,f"{tag}_rep2.bw")
    result_path = os.path.join(data_path,f"{tag}.bw")
    average_bigwig(rep1_path, rep2_path, result_path)