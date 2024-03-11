import torch 
import torch.nn as nn
import sys

class DNN(nn.ModuleDict):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(DNN, self).__init__()

        self.layer_dim = layer_dim

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_ = nn.Linear(60, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        for _ in range(self.layer_dim):
            x = self.activation(self.linear(x))

        out = torch.transpose(x, 1, 2)
        out = self.linear_(out)
        out = torch.transpose(out, 1 ,2)
        out = self.fc(out)
        out = torch.reshape(out, (-1, 2))
        out = self.logsoftmax(out)
        return out
    
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first = True, 
                            dropout = dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        out1 = self.logsoftmax(out)

        return out1
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first = True, 
                            dropout = dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        out, (hn, cn) = self.gru(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        out1 = self.logsoftmax(out)

        return out1

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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        out1 = self.logsoftmax(out)
        return out1
    
class LSTM_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batchNorm = nn.BatchNorm1d(num_features=input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True, 
                            dropout = dropout_prob)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=12, 
                                                   dim_feedforward= 512, dropout=dropout_prob,
                                                   batch_first= True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        decoer_layer = nn.TransformerDecoderLayer(d_model = hidden_dim, n_head = 8,
                                                  dim_feedforward=512, dropout = dropout_prob,
                                                  batch_first= True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoer_layer, num_layers = 12)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Use LSTM first to extract important info for each vector in the sequence
        x = self.batchNorm(x)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        # Last layer hidden out stored in lstm structure
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Attention mechanism to assign weights to each hidden vector, and calculate sequence contribution
        encoder_out = self.encoder(out)
        # Finally, decoder layer to decode encoder output
        # x = target, output from encoder layer
        out = self.decoder(x, encoder_out)

        out = self.fc(out[:,-1,:])
        out1 = self.logsoftmax(out)
        return out1
        
def get_model(model, model_params):
    models = {
        "dnn": DNN,
        "gru": GRU,
        "rnn": RNN,
        'lstm':LSTMModel,
        "lstm_transformer": LSTM_Transformer
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