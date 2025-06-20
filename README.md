# DanQ-based-epigenetic-prediction-of-botanical-genome
DanQ is a hybrid convolutional and recurrent neural network model for predicting the function of DNA de novo from sequence proposed by Quang, D. and Xie, X. Compared to orginial DanQ, we changed the scale of the model to adapt plant DeepSEA data sets, introduce batch normalization layers and L2 regularization.

![image](https://github.com/user-attachments/assets/a1b704ee-b442-4dec-85c3-59031063ce18)


# 1. INSTALL
```
$ conda create -n danq_env python=3.8
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

|            | Epochs | Batch Size | Learning Rate |
|------------|--------|------------|----------------|
| *O. sativa*   | 25     | 122        | 0.00010880242882938465          |
| *Z. mays*      | 20     | 80         | 8.674559776324777e-05         |
| *A. thaliana*  | 28     | 67        | 0.0003748321662847933         |


# 3.Evaluate your model

```
python DanQ_test.py
```

#
Please view the Original_DanQ folder if you need classic DanQ mdeol which is compatiable with python 3.8

