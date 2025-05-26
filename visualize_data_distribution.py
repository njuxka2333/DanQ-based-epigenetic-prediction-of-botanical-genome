import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

trainmat = h5py.File('data/train.mat', 'r')
validmat = h5py.File('data/valid.mat', 'r')
testmat = h5py.File('data/test.mat', 'r')

X_train, y_train = np.array(trainmat['trainxdata']), np.array(trainmat['traindata'])
valid_x, valid_y = np.array(validmat['validxdata']), np.array(validmat['validdata'])


label = []
with open('original_data/tag_osa.txt','r') as file:
    for line in file:
        label.append(line.strip())
label_num = len(label)

train_count,valid_count = {tag:0 for tag in label},{tag:0 for tag in label}

for train_label in tqdm(y_train):
    for i in range(label_num):
        if train_label[i] == 1:
            train_count[label[i]] += 1

for valid_label in tqdm(valid_y):
    for i in range(label_num):
        if valid_label[i] == 1:
            valid_count[label[i]] += 1

train_total,valid_total = y_train.shape[0],valid_y.shape[0]
train_values =[i/y_train.shape[0] for i in list(train_count.values())]
valid_values =[i/valid_y.shape[0] for i in list(valid_count.values())]

plt.plot([i - 0.2 for i in range(len(label))], train_values, marker='o', linestyle='-')
plt.plot([i + 0.2 for i in range(len(label))], valid_values, marker='s', linestyle='--')

plt.xlabel('tags')
plt.title('Comparison of y_train and valid_y')
plt.xticks(range(len(label)), label,rotation=90)
plt.legend()
plt.savefig('Comparison of y_train and valid_y.png')

print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('valid_x:',valid_x.shape)
print('valid_y:',valid_y.shape)