import torch
import sys
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from random import shuffle
import random

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def data(args):
    labels = pd.read_csv('../data/train_labels.csv')
    df = pd.read_csv(args.data_path, index_col =0)
    df = df.merge(labels, on = 'sequence', how = 'left')
 
    train_id = pd.read_csv('../data/train_id.csv')
    val_id = pd.read_csv('../data/val_id.csv')
    train_df = df[df['sequence'].isin(train_id.sequence)]
    val_df = df[df['sequence'].isin(val_id.sequence)]

    not_features = ['subject', 'step', 'sequence', 'state']
    features = [f for f in df.columns if f not in not_features]
    target = ['state']

    scaler = StandardScaler()
    train_df_scal = pd.DataFrame(scaler.fit_transform(train_df[features]), columns = features)
    train_df_scal['state'] = train_df['state'].values
    train_df_scal['sequence'] = train_df['sequence'].values
    val_df_scal = pd.DataFrame(scaler.transform(val_df[features]), columns = features)
    val_df_scal['state'] = val_df['state'].values
    val_df_scal['sequence'] = val_df['sequence'].values

    # print(train_df[features].describe())
    # print(train_df_scal[features].describe())
    # print(val_df_scal[features].describe())
    # sys.exit()

    X, Y = [], []
    X_val, Y_val = [], []

    for i in train_df_scal.sequence.unique():
        X.append(train_df_scal[train_df_scal['sequence'] == i][features].values)
        Y.append(train_df_scal[train_df_scal['sequence'] == i]['state'][-1:].values)

    for i in val_df_scal.sequence.unique():
        X_val.append(val_df_scal[val_df_scal['sequence'] == i][features].values)
        Y_val.append(val_df_scal[val_df_scal['sequence'] == i]['state'][-1:].values)


    return X, Y, X_val, Y_val, scaler


class CustomDataset(Dataset):
    def __init__(self, target, x_input):
        self.target = target
        self.x_input = x_input

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        y = self.target[index]
        x = self.x_input[index]
        return x, y
    
def build_datasets(X_train, Y_train, X_val, Y_val, args):

    train_data = CustomDataset(Y_train, X_train)
    train_dataloader = DataLoader(train_data, args.batch_size)

    val_data = CustomDataset(Y_val, X_val)
    val_dataloader = DataLoader(val_data, args.batch_size)

    return train_dataloader, val_dataloader








    


    
