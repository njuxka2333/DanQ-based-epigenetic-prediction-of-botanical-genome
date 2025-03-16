# DanQ-based-epigenetic-prediction-of-botanical-genome

# 1. Build DeepSEA data for model training

```
$ conda create -n danq_env python=3.8
$ pip install numpy scipy Biopython h5py tensorflow-gpu
$ conda activate danq_env
```
After activating the python enviorment and setting up, perform build_DeepSEA_data.py based on your original data(you can find sample data in data file).Finally，there are training dataset, valid dataset and test dataset with the form of .mat in the data file.

```
$ mkdir data
$ python build_DeepSEA_data.py\
--label_filename original_data/ath/tag_ath.txt \
--original_train_data_filename original_data/ath/mergedtag_ath_1024_512.fa \
--original_test_data_filename original_data/ath/mergedtag_ath_1024_500.fa \
--train_filename data/ath_data/train.mat \
--valid_filename data/ath_data/valid.mat \
--test_filename data/ath_data/test.mat \
--train_data_filename data/ath_data/train_data.npy \
--train_labels_filename data/ath_data/train_labels.npy \
--valid_data_filename data/ath_data/valid_data.npy \
--valid_labels_filename data/ath_data/valid_labels.npy \
--test_data_filename data/ath_data/test_data.npy \
--test_labels_filename data/ath_data/test_labels.npy
```

# 2. Train your model
```
$ python DanQ_train_refreshed.py\
--train_filepath data/ath_data/train.mat \
--valid_filepath data/ath_data/valid.mat \
--test_filepath data/ath_data/test.mat \
--output_model_filepath DanQ_bestmodel.h5 \
--output_loss_curve_filepath training_valid_loss_curve.png
```
Or
```
$ python DanQ_train_with_hyperparameter.py\
--train_filepath data/ath_data/train.mat \
--valid_filepath data/ath_data/valid.mat \
--test_filepath data/ath_data/test.mat \
--output_model_filepath DanQ_bestmodel.h5 \
--output_loss_curve_filepath training_valid_loss_curve.png
```

# 3.Evaluate your model
```
$ python DanQ_test_refreshed.py \
—test_file path data/osa_data/test.mat \
—model DanQ_bestmodel_osa.hdf5 \
—roc_curve roc_curve_osa.png \
—prauc_curve_osa.png \
```

