"""Build the DeepSEA dataset."""
import argparse
import logging
import h5py
import numpy as np
from Bio import SeqIO

logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument("--label_filename",help="source  label file")
parser.add_argument("--original_train_data_filename",help="train data genome file")
parser.add_argument("--original_test_data_filename",help="test data genome file")
parser.add_argument("--train_filename", help="Output train dataset filename (.mat format)")
parser.add_argument("--valid_filename", help="Output valid dataset filename (.mat format)" )
parser.add_argument("--test_filename", help="Output test dataset filename (.mat format)")
parser.add_argument("--train_data_filename", help="Output train data filename (.npy format)")
parser.add_argument("--train_labels_filename", help="Output train labels filename (.npy format)")
parser.add_argument("--valid_data_filename", help="Output valid data filename (.npy format)")
parser.add_argument("--valid_labels_filename", help="Output valid labels filename (.npy format)")
parser.add_argument("--test_data_filename", help="Output test data filename (.npy format)")
parser.add_argument("--test_labels_filename", help="Output test labels filename (.npy format)" )

args = parser.parse_args()

def extract_label(s,label):
    l= np.zeros(len(label),dtype=np.int8)
    i = 0
    for x in label:
        if x in s:
            l[i] = 1
        i += 1
    return(l)

def generate_data(s):
    data = np.zeros((1024,4),dtype=np.int8)
    for i in range(0,1024):
        if s[i] == 'A':
            data[i,0] = 1
        elif s[i] == 'G':
            data[i,1] = 1
        elif s[i] =='C':
            data[i,2] =1
        elif s[i] == 'T':
            data[i,3] =1
    return data

def save_to_mat(filename, data, labels, key):
    """Save data and labels to .mat file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset(f"{key}xdata",  data=data)
        f.create_dataset(f"{key}data",  data=labels)

label_path =args.label_filename
label = []
with open(label_path, 'r') as file:
    for line in file:
        label.append(line.strip())

train_vaild_list = []
train_vaild_label_list = []
train_path = args.original_train_data_filename
train_valid_num = 1
for record in SeqIO.parse(train_path, 'fasta'):
    train_vaild_label_list.append(extract_label(record.id,label))
    train_vaild_list.append(generate_data(str(record.seq)))
    print('第{}条训练记录'.format(train_valid_num))
    train_valid_num += 1

split_index = int(len(train_vaild_list) * 0.95)
train_data = train_vaild_list[:split_index]
train_labels =train_vaild_label_list[:split_index]
valid_data = train_vaild_list[split_index:]
valid_labels = train_vaild_label_list[split_index:]
print('{}条训练序列'.format(len(train_data)))
print('{}条训练标签'.format(len(train_labels)))
print('{}条验证序列'.format(len(valid_data)))
print('{}条验证标签'.format(len(valid_labels)))

train_data = np.stack(train_data, axis=0)
train_labels = np.stack(train_labels, axis=0)
valid_data = np.stack(valid_data, axis=0)
valid_labels = np.stack(valid_labels, axis=0)

test_list = []
test_label_list = []
test_path = args.original_test_data_filename
test_num = 1
for record in SeqIO.parse(test_path, 'fasta'):
    if len(record.seq) >= 1024:
        test_label_list.append(extract_label(record.id,label))
        test_list.append(generate_data(str(record.seq)))
        print('第{}条测试记录'.format(test_num))
        test_num += 1

test_data = np.stack(test_list, axis=0)
test_labels = np.stack(test_label_list, axis=0)
print('{}条测试序列'.format(len(test_data)))
print('{}条测试标签'.format(len(test_labels)))

if args.train_filename:
    save_to_mat(args.train_filename, train_data, train_labels, "train")
    print('训练文件保存成功')
if args.valid_filename:
    save_to_mat(args.valid_filename, valid_data, valid_labels, "valid")
    print('验证文件保存成功')
if args.test_filename:
    save_to_mat(args.test_filename, test_data, test_labels, "test")
    print('测试文件训练成功')

if args.train_data_filename:
    np.save(args.train_data_filename, train_data)
    print('训练序列保存成功')
if args.train_labels_filename:
    np.save(args.train_labels_filename, train_labels)
    print('训练标签保存成功')
if args.valid_data_filename: 
    np.save(args.valid_data_filename, valid_data)
    print('验证序列保存成功')
if args.valid_labels_filename:
    np.save(args.valid_labels_filename, valid_labels)
    print('验证标签保存成功')
if args.test_data_filename:
    np.save(args.test_data_filename, test_data)
    print('测试序列保存成功')
if args.test_labels_filename:
    np.save(args.test_labels_filename, test_labels)
    print('测试标签保存成功')
