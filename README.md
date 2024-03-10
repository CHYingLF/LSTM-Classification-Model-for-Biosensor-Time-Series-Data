# Descriptiom
A hands-on implementation of RNN/GRU/LSTM/TRansformer model for sequencce data classification (or regression), especially time series data with local GPU.

**Model**

Recurrent Neural Network (RNN) has been used a lot for sequence input data, expecially natural language processing, because it can keep the information along the sequence for final output. Later, gate neural network (GNN), long-short-term-memory (LSTM) have been published to improve the RNN structure. They include somes gates to intentially keep the important long range information and forget some unimportant ones. Long Short-Term Memory (LSTM) is a specialized type of recurrent neural network (RNN) architecture designed to model sequential data and capture long-range dependencies effectively. The LSTM cell illustrates the information flow for x input:
 
Where x_t: input feature at time t, h_t: hidden state, c_t  : cell state, f_t : forget gate, i_t : input gate, o_t: output gate, σ : activation function, W: weight matrix. In our model, the activation function is sigmoid function. And we used a stack LSTM model structure, where 5 layers of LSTM cells are stacked together, and the hidden state size is 32.

Gates are critical components responsible for controlling the flow of information within the network and regulating the interaction between different parts of the memory cell. These gates include the input gate, forget gate, and output gate, each serving a specific purpose in managing the information flow through the network.
Input Gate (i_t): The input gate determines how much new information will be stored in the cell state at the current time step. It takes input from the current input data (x_t) and the previous hidden state (h_t). Through a sigmoid activation function, the input gate decides which values from the input should be updated and added to the cell state.

Forget Gate (f_t): The forget gate controls the extent to which the previous cell state (c_(t-1)) should be retained or forgotten. It considers the input data (x_t) and the previous hidden state (h_(t-1)) and, through a sigmoid activation function, determines which information from the previous cell state should be discarded.
Output Gate (o_t): The output gate regulates how much information from the current cell state (c_t) should be exposed to the network's output at the current time step. It considers the current input data (x_t) and the previous hidden state (h_(t-1)), along with the updated cell state (c_t), and passes this information through a sigmoid activation function to determine the output activation.

Transformer model has populated to every ML region since it has been proven to be so powerful in natual language processing, especially after ChatGPT has published in 2022. Compare to LSTM, the implementation of Transformer is not easy as it requires siginificantly more data to train. Despite this, the attention mechanism in transformer model is so promising to assign weights to each sequence. Here, we combine the LSTM model output with attention block, where the final hidden output is properly processed to give a good results.

**Features for this Repo**

0. Ideally can be used for any sequence input with any number of features.
1. Aiming to solve varies length input of time series data, capture local and global information using LSTM/Transformer model.
2. Also implement attention mechanism to assign attention score on the last hidden layer output of LSTM flow.
3. If local GPU is available for ML traning, can follow the steps to setup the GPU for local traning. Own your ML GPU, forget about paying online GPU to train.
4. The final R2 score is 0.88 on the validation dataset of biological sensors' data.


# Environemnt
python 3.8\
Pytorh 2.1.2\
cuda 11.1\
numpy 1.24.3\
pandas 2.0.3\
scikit-learn 1.3.2\
matplotlib 3.7.2\
seaborn 0.12.2\
tqdm 4.65.0

# GPU Setup Guide
1. First we need find out if your local computer has a Nvidia GPU that can be used for ML traning and Download the appropriate driver. (1) Search 'Device Manager', under 'Display Adapter', we can see the available GPU. (2) Check your if GPU name is listed on Nvidia ML-enabled GPU website (3) Or directly go to GPU driver download page to see if there is an available driver for your GPU: https://www.nvidia.com/Download/index.aspx
2. Install Anaconda. Download anoconda/miniconda if not so: https://www.anaconda.com/
3. Create new conda environment for gpu training. 'Search and open anaconda prompt', in the terminal, type  ```conda create -n py38_gpu python=3.8```
4. Install gpu enabled PyTorch. Visit https://pytorch.org/, select you OS and and CUDA version. Be sure to check the cuda version suited for you GPU.
5. Install Cuda. https://developer.nvidia.com/cuda-11-7-0-download-archive. Be sure to check the proper CUDA version for your GPU name
6. Verify CUDA with the pytorch: Open python file and run following
```
import torch
print(torch.cuda.is_available())
```
if 'True' is printed, then congraduration, you have you own ML GPU. If not, do check previous steps, especially if you gpu name, driver version, pytorch version, CUDA version are properly compatible.

7. User Visual Studio Code to run this code repo. After everything is ready, open Visual studio code, and opent the downloaded code repo dir, change 'device' to 'gpu', select the gpu conda environment by clicking the right bottom python version. Or go to top panel Terminal/New terminal/, and then type "Conda activate Py38_gpu", here you can also run the code with "Python main.py"

# Training
*Steps*:
1. Change the the parameters configuration based on your problems in the ```main.py```, or use a yaml file description

Following table summarizes the parameters that may need to be adjusted. Please note each problem can be different, but it is always a good option to start with the default value except for the model input and out dimension, which should be changed to specific problem. After you have a good sence of the training, you can start tunning the parameters to achieve the best performace.

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


2. Once you have the parameters setting adjusted to your problem, and specify necessary values like data_path, val_path, input and out dimension. You can start training by typing ```python train.py```.

# Evaluation

After training, you will see the model weight file ('.pth') in ```./outdir```, excute the evaluation code by ```python eval.py```
