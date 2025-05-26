import h5py
import numpy as np
from Bio import SeqIO

# generate complement sequence
def complement_dna(s):
    complement= {'A':'T','T':'A','C':'G','G':'C'}
    c_s = ""
    for nucleotide in s:
        c_s += complement[nucleotide]
    return c_s

# generate label matrix
def extract_label(s,label):
    l= np.zeros(len(label),dtype=np.int8)
    i = 0
    for x in label:
        if x in s:
            l[i] = 1
        i += 1
    return(l)

# generate 1024x4 matrix for DNA sequence
def generate_data(s):
    data = np.zeros((1024,4),dtype=np.int8)
    for i in range(0,1024):
        if s[i] == 'A':
            data[i,0] = 1
        elif s[i] == 'G':
            data[i,1] = 1
        elif s[i] =='C':
            data[i,2] = 1
        elif s[i] == 'T':
            data[i,3] = 1
    return data

# save data and label to  a .mat file, _xdata with data and _data with label
def save_to_mat(filename, data, labels, key):
    """Save data and labels to .mat file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset(f"{key}xdata",  data=data)
        f.create_dataset(f"{key}data",  data=labels)
