# Descriptiom
A implementation of RNN/GRU/LSTM/TRansformer model for sequencce data classification (or regression), especially time series data.

Features:
0. Ideally can be used for any sequence input with any number of features.
1. Aiming to solve varies length input of time series data, capture local and global information using LSTM model.
2. Also implement attention mechanism to assign attention score on the last hidden layer output of LSTM flow
3. The final R2 score is 0.88 on the validation dataset of biological sensors' data.

# Environemnt
python 3.8
Pytorh 2.1.2
cuda 11.1
numpy 1.24.3
pandas 2.0.3
scikit-learn 1.3.2
matplotlib 3.7.2
seaborn 0.12.2
tqdm 4.65.0

# Training
## Parameters description
| Parameters | Explanation | Suggested Values|
| :---:      | :---:       | :---:           |
| lr_start   | start learning rate | 1e-5    |


# Evaluation
