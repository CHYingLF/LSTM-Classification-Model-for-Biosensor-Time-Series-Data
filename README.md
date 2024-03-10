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
Steps:
1. Change the the parameters configuration based on your problems in the '''main.py''', or use a yaml file description

Followng table summariezd the parameters that may need to adjusted, note each problem can be different, but it is always a good option to start with the default value except for the model input and out dimension, which should be changed to specific problem.

| Parameters | Explanation | Suggested Values|
| :---:      | :---:       | :---:           |
| lr_start   | start learning rate | 1e-3    |
| lr_base    | base for exponetial decay of learning rate. should be in [0.1, 0.99], smaller then lr decay fast with respect to epochs | from 0.95 to 0.99|
|lr_end|the smallest lr that the traning can use | 1e-6 |
|weight_decay|L2 regularization for AdamW optimization, intend to avoid overfitting|1e-3 for no weaker overfitting, 1e-1 for strong overfitting|
|batch_size| number of samples in one batch| 2,4,8,16|
|n_epochs|number of epoch for training|100-500|
|input_dim| number of the features| None|
|hidden_dim| hidden dim of the LSTM structure| 32|
|layer_dim| number of stacked LSTM layers | 5|
|output_dim| number of output values, 1 for single target regression| 1|
|dropout_prob| dropout probability for the LSTM/transformer/DNN neurons, can allleviate overfitting. 0 for no dropout| 0|
|device| 'gpu' for gpu training, need to configure your GPU with CUDA first; 'cpu' for cpu training|cpu|
|outdir|dir for output results| './outdir'|
|early_stop_roounds|number of consective rounds that loss is not decreasing, traning stops|10|
|data_path|path of your traning data source, can accept '.csv' file|''|
|val_path| path of you val data file|''|
|random_seed|random seed to shuffle data, initialize weight, control repeatibility of the training|666|
|model_name|choose the model you want to use, options are: 'lstm', 'rnn', 'dnn', 'gru', 'lstm_transformer'|'lstm'|




# Evaluation
