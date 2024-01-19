import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import datetime
from datasets.data_processor import data, build_datasets
from models.backbone import build_model
from engine import Optimization
#from utils.metrics import calculate_metrics, format_prediction

def get_args_parse():
    parser = argparse.ArgumentParser("Set model", add_help = False)
    parser.add_argument('--lr_start', default = 2e-4, type = float)
    parser.add_argument('--lr_base', default = 0.98, type = float)
    parser.add_argument('--lr_end', default = 1e-5, type = float)
    parser.add_argument('--weight_decay', default=0.001, type = float)
    parser.add_argument('--batch_size', default = 4, type = float)
    parser.add_argument('--n_epochs', default =100, type=int)
    parser.add_argument('--input_dim', default = 13, type = int)
    parser.add_argument('--hidden_dim', default =32, type = int)
    parser.add_argument('--layer_dim', default = 5, type = int)
    parser.add_argument('--output_dim', default = 2, type = int)
    parser.add_argument('--dropout_prob', default = 0., type = float)
    parser.add_argument('--device', default = 'cpu', type = str)
    parser.add_argument('--outdir', default = './outdir/', type = str)
    parser.add_argument('--early_stop_rounds', default = 10, type = int)
    parser.add_argument('--data_path', default = '../data/train.csv', type = str)
    parser.add_argument('--val_path', default = '../data/', type = str)
    parser.add_argument('--random_seed', default = 44, type = int)
    parser.add_argument('--model_name', default='lstm', type = str)

    return parser

def main(args):
    time0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device} is available")

    # set randome seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # data
    X, Y, X_val, Y_val, _ = data(args)

    # build data loader
    train_dataloader, val_dataloader = build_datasets(X, Y, X_val, Y_val, args)

    train_x, train_y = next(iter(train_dataloader))
    print("Feature bacth shape:", train_x.size())
    print("Labels batch shape:", train_y.size())
    print("Total number of sequence for training:", len(train_dataloader))
    print("Total number of sequence for validation", len(val_dataloader))

    # build model
    model = build_model(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of training paramters", total_params)

    # build optimization
    weight = torch.tensor([1.0, 1.0]).to(args.device)
    loss_fn = nn.NLLLoss(weight=weight, reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=args.weight_decay)
    opt = Optimization(model=model.to(device), loss_fn=loss_fn, optimizer=optimizer, args=args)

    # train
    print("Start training:")
    opt.train(train_dataloader, val_dataloader, batch_size=args.batch_size, n_epochs=args.n_epochs, n_features=args.input_dim)
    opt.plot_losses(args.outdir)

    print("Total time used:", '%.2f min'%((time.time()-time0)/60))  

if __name__ == '__main__':
    pargs = argparse.ArgumentParser("Biology state", parents = [get_args_parse()])
    args = pargs.parse_args()
    main(args)


    



