import torch 
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True, 
                            dropout = dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print('x:',x)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        out1 = self.logsoftmax(out)
        #print('out',out1)
        #print(out1)

        return out1
        
def get_model(model, model_params):
    models = {
        'lstm':LSTMModel
    }

    return models.get(model.lower())(**model_params)


def build_model(args):
    model_params = {'input_dim':args.input_dim,
                    'hidden_dim': args.hidden_dim,
                    'layer_dim': args.layer_dim,
                    'output_dim': args.output_dim,
                    'dropout_prob': args.dropout_prob}
    model = get_model(args.model_name, model_params)

    return model