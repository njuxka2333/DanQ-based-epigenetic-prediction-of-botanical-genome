"""Build the DeepSEA dataset."""
import random
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from process_data import extract_label,generate_data,save_to_mat, complement_dna
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# get training-valid data, test data and the label file
training_valid_record = list(SeqIO.parse('data_fuxianzhao/osa/mergedtag_osa_1024_512.fa', 'fasta'))
test_record = list(SeqIO.parse('data_fuxianzhao/osa/mergedtag_osa_1024_500.fa','fasta'))

# get the label file
labels = []
with open('data_fuxianzhao/osa/tag_osa_new.txt','r') as file:
    for line in file:
        labels.append(line.strip())

# generate training and valid data
train_data,train_labels = [],[]
valid_data,valid_labels = [],[]
valid_size = int(len(training_valid_record)*2*0.05) # 5% of dataset for validation

for record in tqdm(training_valid_record,desc="Processing training and valid data"):
    if len(record.seq) == 1024:
        chrom = record.id.split('::')[1].split(':')[0] # get chromosome name
        label = extract_label(record.id,labels) # generate label matrix
        sequence = str(record.seq)  #get sequence
        complement_sequence = complement_dna(sequence) # generate complement sequence
        if np.any(label):
            if len(valid_data) < valid_size and chrom == '12':  # select a chromosome for validation
                valid_data.append(generate_data(sequence))
                valid_labels.append(label)
                valid_data.append(generate_data(complement_sequence))
                valid_labels.append(label)
            else:
                train_data.append(generate_data(sequence))
                train_labels.append(label)
                train_data.append(generate_data(complement_sequence))
                train_labels.append(label)

train_data = np.stack(train_data,axis=0)
train_labels =np.stack(train_labels,axis=0)

valid_data = np.stack(valid_data,axis=0)
valid_labels = np.stack(valid_labels,axis=0)

print('{} train sequences'.format(len(train_data)))
print('{} train labels'.format(len(train_labels)))
print('{} valid sequences'.format(len(valid_data)))
print('{} valid labels'.format(len(valid_labels)))

# generate test data
test_data = []
test_labels = []
for record in tqdm(test_record,desc="Processing test data"):
    if len(record.seq) == 1024:
        label = extract_label(record.id,labels)
        sequence = str(record.seq)
        complement_sequence = complement_dna(sequence)
        if np.any(label):
            test_data.append(generate_data(sequence))
            test_labels.append(label)
            test_data.append(generate_data(complement_sequence))
            test_labels.append(label)

test_data = np.stack(test_data, axis=0)
test_labels = np.stack(test_labels, axis=0)
print('{} test sequences'.format(len(test_data)))
print('{} test labels'.format(len(test_labels)))


# save training dataset,valid dataset and test dataset to train.mat, valid.mat and test.mat respectively
save_to_mat('fuxianzhao/data/osa_data/train.mat', train_data, train_labels, "train")
print('training file saved ')
save_to_mat('fuxianzhao/data/osa_data/valid.mat', valid_data, valid_labels, "valid")
print('valid file saved')
save_to_mat('fuxianzhao/data/osa_data/test.mat', test_data, test_labels, "test")
print('test file saved')


