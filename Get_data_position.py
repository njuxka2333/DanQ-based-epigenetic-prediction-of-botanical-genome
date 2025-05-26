"""Build the DeepSEA dataset."""
import random
import numpy as np
import pandas as pd
from Bio import SeqIO



# get training-valid data, test data and the label file
test_record = list(SeqIO.parse('/Users/morgen_zhao/Desktop/Thesis/orignial_data/zma/mergedtag_zma_1024_500.fa','fasta'))

df = []
for record in test_record:
    if len(record.seq) == 1024:
        seq = {}
        seq['chrom'],seq['start'],seq['end'] = int(record.id.split('::')[1].split(':')[0]),int(record.id.split('::')[1].split(':')[1].split('-')[0]),int(record.id.split('::')[1].split(':')[1].split('-')[1])
        df.append(seq)


df = pd.DataFrame(df).sort_values(by=["chrom", "start"]).reset_index(drop=True)
df.to_csv("zma_1024_500_position.csv")
print("zma_1024_500_position csv saved")

    
