from datasets.data_processor import data, build_datasets
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from models.backbone import build_model
import numpy as np
import argparse
import torch

def get_args_parse():
    parser = argparse.ArgumentParser("Set model", add_help = False)
    parser.add_argument('--lr_start', default = 2e-4, type = float)
    parser.add_argument('--lr_base', default = 0.98, type = float)
    parser.add_argument('--lr_end', default = 1e-5, type = float)
    parser.add_argument('--weight_decay', default=0.001, type = float)
    parser.add_argument('--batch_size', default = 4, type = float)
    parser.add_argument('--n_epochs', default =500, type=int)
    parser.add_argument('--input_dim', default = 13, type = int)
    parser.add_argument('--hidden_dim', default =64, type = int)
    parser.add_argument('--layer_dim', default = 4, type = int)
    parser.add_argument('--output_dim', default = 2, type = int)
    parser.add_argument('--dropout_prob', default = 0., type = float)
    parser.add_argument('--device', default = 'cpu', type = str)
    parser.add_argument('--outdir', default = './outdir/', type = str)
    parser.add_argument('--early_stop_rounds', default = 10, type = int)
    parser.add_argument('--data_path', default = '../data/train.csv', type = str)
    parser.add_argument('--val_path', default = '../data/', type = str)
    parser.add_argument('--random_seed', default = 44, type = int)
    parser.add_argument('--model_name', default='lstm', type = str)
    parser.add_argument('--model_path', default = './outdir/lstm_64_4_weight.pth', type = str)

    return parser


def main(args):
    # data
    X, Y, X_val, Y_val, _ = data(args)

    # build data loader
    _, val_dataloader = build_datasets(X, Y, X_val, Y_val, args)

    # load model
    model = build_model(args)
    model.load_state_dict(torch.load(args.model_path))

    y_true, y_pred = [], []
    with torch.no_grad():
        for x_val, y_val in val_dataloader:
            x_val = x_val.to(torch.float32).to(args.device)
            y_val = y_val.to(args.device)
            #print(type(y_val))
            y_true.extend([y.item() for y in y_val])
            #print(y_true)
            model.eval()
            yhat = model(x_val)
            yhat_ = np.argmax(yhat, axis = 1)
            y_pred.extend([y.item() for y in yhat_])
            #val_loss = self.loss_fn(yhat, y_val.squeeze(dim=1)).item()
            #batch_val_loss.append(val_loss)

        #validation_loss = np.mean(batch_val_loss)
        #self.val_loss.append(validation_loss)

    #print(y_true, y_pred)
    precision = '%.2f'%(precision_score(y_true, y_pred))
    recall = '%.2f'%(recall_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    print(precision, recall)
    print(cm)



if __name__ == '__main__':
    pargs = argparse.ArgumentParser("Biology state", parents = [get_args_parse()])
    args = pargs.parse_args()
    main(args)

