# Build mini data and mini training
# for configer
import os
import logging
import scipy.io
from Bio import SeqIO
import numpy as np
import pandas as pd
from process_data import extract_label,generate_data
from DanQ_model import DataGenerator, create_model, LoggingCallback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, r2_score, f1_score

# only use one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set NCCL envioroment
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_DISABLE'] = '1'


# get the fasta file and the label file
label = []
with open(args.label_filename, 'r') as file:
    for line in file:
        label.append(line.strip())

all_records = list(SeqIO.parse(args.original_data_filename, 'fasta'))
all_num = len(all_records)
all_chrom = ['1','2','3','4','5'] # find them in the fasta file

training_losses, valid_losses, test_losses,test_aucs, r2, f1 = dict(),dict(),dict(),dict(),dict(),dict()
df = {}
paired_chroms = [('1','3'),('2','4')('5','6'),('9','10'),('11','12'),('7','8')] # every paried chromosomes are with similar sizes
for chrom1,chrom2 in paired_chroms:
    # valid data from chrom1 and test data from chrom2
    train_data,train_labels = [],[]
    valid_data,valid_labels = [],[]
    test_data,test_labels = [],[]
    valid_num,test_num = all_num*2*0.1,all_num*2*0.1
    for record in all_records:
        chrom = record.id.split('::')[1].split(':')[0]
        if chrom == chrom1 :
            valid_data.append(generate_data(str(record.seq)))
            valid_labels.append(extract_label(record.id, label))
        elif chrom == chrom2:
            test_data.append(generate_data(str(record.seq)))
            test_labels(extract_label(record.id, label))
        else:
            train_data.append(generate_data(str(record.seq)))
            train_labels(extract_label(record.id, label))

    X_train,y_train = np.stack(train_data, axis=0),np.stack(train_labels, axis=0)
    valid_x,valid_y = np.stack(valid_data, axis=0),np.stack(valid_labels, axis=0)
    test_x,test_y = np.stack(test_data, axis=0),np.stack(test_labels, axis=0)
    
    # fit model with chrom 1 of valid dataset and chrom2 of test dataset
    model = create_model(lr=1e-3)  # lr
    batch_size = 121  
    # Data Generator
    train_generator = DataGenerator(X_train, y_train, batch_size=batch_size, shuffle=True, normalize=True, augment_rc=True)
    val_generator = DataGenerator(valid_x, valid_y, batch_size=batch_size, shuffle=True, normalize=True, augment_rc=True)

    # checkpointer
    checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel_fold.hdf5", verbose=1, save_best_only=True)

    #earlystopper
    earlystopper = EarlyStopping(monitor='val_dice_score', patience=3, verbose=1) 

    #logging callback
    logging_callback = LoggingCallback()

    # training model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=8,  # epoch number
        callbacks=[checkpointer,logging_callback],
        workers=1,
        use_multiprocessing=False
    )
    
    #calculate test loss R square and f1 score
    test_loss, test_auc= model.evaluate(test_x, test_y)
    y_pred = model.predict(test_x)
    y_pred_binary = np.round(y_pred)  # 将预测值转换为二进制
    r2 = r2_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred_binary, average='micro')

    # record test results of chrom 1 and chrom2
    df[chrom1+'-'+chrom2] = {
                            "test loss": test_loss,
                            "r2_square": r2 ,
                            "f1_score": f1
    }
    
    # exchange valid dataset and test data set
    valid_x,test_x = test_x,valid_x
    valid_y, test_y = test_y,valid_y
    
    # fit model with chrom 2 of valid dataset and chrom1 of test dataset
    #generate data generator again
    val_generator = DataGenerator(valid_x, valid_y, batch_size=batch_size, shuffle=True, normalize=True, augment_rc=True)

    # training model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=8,  # epoch number
        callbacks=[checkpointer,logging_callback],
        workers=1,
        use_multiprocessing=False
    )
    
    #calculate test loss R square and f1 score
    test_loss, test_auc= model.evaluate(test_x, test_y)
    y_pred = model.predict(test_x)
    y_pred_binary = np.round(y_pred)  # 将预测值转换为二进制
    r2 = r2_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred_binary, average='micro')

    # record test results of chrom 2 and chrom1
    df[chrom2+'-'+chrom1] = {
                            "test loss": test_loss,
                            "r2_square": r2 ,
                            "f1_score": f1
    }

df = pd.Dataframe(df)
df.to excel("chrom_pos.xlsx")

    



    



