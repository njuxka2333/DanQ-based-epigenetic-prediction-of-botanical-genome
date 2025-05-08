# DanQ-based-epigenetic-prediction-of-botanical-genome

# 1. Build DeepSEA data for model training

```
$ conda create -n danq_env python=3.8
$ pip install numpy pandas matplotlib scipy Biopython h5py scikit-learn scikit-optimize tensorflow-gpu==2.8.0
$ conda activate danq_env
```
After activating the python enviorment and setting up, perform build_DeepSEA_data.py based on your original data(you can find sample data in data file).Finally，there are training dataset, valid dataset and test dataset with the form of .mat in the data file.

# 2. Train your model

# 3.Evaluate your model


