import sys
import torch
import matplotlib.pyplot as plt
import copy
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score
#from utils.metrics import calculate_metrics, format_prediction

class Optimization():
    def __init__(self, model, loss_fn, optimizer, args):
        self.model = model
        self.loss_fn = loss_fn
        self.optmizer = optimizer
        self.train_losses = []
        self.val_loss = []
        self.args = args
        self.best_metrics = 0

    def train_step(self, x, y):
        '''The method train_step completes one step of training
        '''
        # Sets model to train mode
        self.model.train()
        # Makes prediction
        yhat = self.model(x)
        # Computes loss
        loss = self.loss_fn(yhat, y)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeros gradients
        self.optmizer.step()
        self.optmizer.zero_grad()

        return loss.item()
    
    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = self.args.outdir+ f'{self.args.model_name}_{self.args.hidden_dim}_{self.args.layer_dim}_weight.pth'
        best_model_path = self.args.outdir + f'{self.args.model_name}_{self.args.hidden_dim}_{self.args.layer_dim}_best_model.pth'
        counter = 0
        best_r2 = -1

        for epoch in range(1, n_epochs+1):
            self.optmizer.param_groups[0]['lr'] = max(self.args.lr_start * self.args.lr_base**(epoch-1), self.args.lr_end)
            time1 = time.time()
            
            batch_losses = []
            for x_batch, y_batch in train_loader:
                #print(x_batch.shape)
                x_batch = x_batch.to(torch.float32).to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                #print('y:',y_batch)
                loss = self.train_step(x_batch, y_batch.squeeze(dim = 1))
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            y_true, y_pred = [], []
            with torch.no_grad():
                batch_val_loss = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(torch.float32).to(self.args.device)
                    y_val = y_val.to(self.args.device)
                    #print(type(y_val))
                    y_true.extend([y.item() for y in y_val])
                    #print(y_true)
                    self.model.eval()
                    yhat = self.model(x_val)
                    yhat_ = np.argmax(yhat, axis = 1)
                    y_pred.extend([y.item() for y in yhat_])
                    val_loss = self.loss_fn(yhat, y_val.squeeze(dim=1)).item()
                    batch_val_loss.append(val_loss)

                validation_loss = np.mean(batch_val_loss)
                self.val_loss.append(validation_loss)

            #print(y_true, y_pred)
            precision = '%.2f'%(precision_score(y_true, y_pred))
            recall = '%.2f'%(recall_score(y_true, y_pred))
            time2 = time.time()
            epoch_time = '%.2f'%((time2 - time1)/60)
            if (epoch%1 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}\t"\
                    f"lr: {self.optmizer.param_groups[0]['lr']:.8f}\t Precision: {precision} Recall: {recall} Epoch time(m):{epoch_time}" 
                )

            # early stop
            if epoch>2:
                if self.val_loss[-1]>self.val_loss[-2]: 
                    counter += 1
                else:
                    counter = 0

            if counter > self.args.early_stop_rounds:
                print("Early stop due to val loss not decrese for ", self.args.early_stop_rounds, 'rounds')
                sys.exit()

            torch.save(self.model.state_dict(), model_path)
            #torch.save(best_model.state_dict(), best_model_path)

    def plot_losses(self, outdir):
        plt.style.use('ggplot')
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.val_loss, label="Validation loss")
        plt.legend()
        plt.title(f"Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(outdir+f'loss_{self.args.hidden_dim}_{self.args.layer_dim}.png', dpi = 300)
        plt.close()

