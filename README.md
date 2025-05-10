# DanQ-based-epigenetic-prediction-of-botanical-genome
DanQ is a hybrid convolutional and recurrent neural network model for predicting the function of DNA de novo from sequence proposed by Quang, D. and Xie, X. Compared to orginial DanQ, we changed the scale of the model to adapt plant DeepSEA data sets, introduce batch normalization layers and L2 regularization.

# 1. INSTALL
```
$ conda create -n danq_env python=3.8
$ pip install numpy pandas matplotlib scipy Biopython h5py scikit-learn scikit-optimize tensorflow-gpu==2.8.0
$ conda activate danq_env
$ pip install biopython h5py numpy matplotlib pandas scikit-learn scikit-optimize scipy tensorflow-gpu tqdm
```

# 2. Build DeepSEA data for model training
We select three model plant species, rice (Oryza sativa), corn (Zea mays) and Arabidopsis thaliana, to construct corresponding DeepSEA  datasets for subsequent species-specific deep learning model training.Unlike the original DeepSEA dataset based on the human genome, this paper does not directly construct training samples from plant genome sequences, but uses preprocessed DNA sequence data.

```
$ python build_DeepSEA_data.py

```
(Optional) Visualize your datasets

```
$ python EDA.py

```

After activating the python enviorment and setting up, perform build_DeepSEA_data.py based on your original data(you can find sample data in data file).Finally，there are training dataset, valid dataset and test dataset with the form of .mat in the data file.

# 2. Train your model

```
python DanQ_train.py
```

# 3.Evaluate your model

```
python DanQ_test.py
```

