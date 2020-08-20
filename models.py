import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #selecting device

class RNNModel(nn.Module):
    def __init__(self,input_size, hidden_size, rnn_layers, fc_hidden_dim):
        super(RNNModel, self).__init__()
        
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.fcl = nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=fc_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=fc_hidden_dim, out_features=input_size)
                    )


    def forward(self, x, prev_state):
        out, state = self.rnn(x, prev_state)
        fcl_out = self.fcl(out.reshape(-1,out.size(2)))
        return fcl_out, state


    def init_hidden(self, batch_size):
        return torch.zeros((self.rnn_layers, batch_size, self.hidden_size), device=device)

    def getModelName(self):
        return "RNN"


class LSTMModel(nn.Module):
    def __init__(self,input_size, hidden_size, lstm_layers, fc_hidden_dim):
        super(LSTMModel, self).__init__()
        
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fcl = nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=fc_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=fc_hidden_dim, out_features=input_size)
                    )


    def forward(self, x, prev_state):
        out, state = self.lstm(x, prev_state)
        fcl_out = self.fcl(out.reshape(-1,out.size(2)))
        return fcl_out, state


    def init_hidden(self, batch_size):
        return [torch.zeros((self.lstm_layers, batch_size, self.hidden_size), device=device), 
                torch.zeros((self.lstm_layers, batch_size, self.hidden_size), device=device)]

    def getModelName(self):
        return "LSTM"


class GRUModel(nn.Module):
    def __init__(self,input_size, hidden_size, gru_layers, fc_hidden_dim):
        super(GRUModel, self).__init__()
        
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=gru_layers, batch_first=True)
        self.fcl = nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=fc_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=fc_hidden_dim, out_features=input_size)
                    )


    def forward(self, x, prev_state):
        out, state = self.gru(x, prev_state)
        fcl_out = self.fcl(out.reshape(-1,out.size(2)))
        return fcl_out, state


    def init_hidden(self, batch_size):
        return torch.zeros((self.gru_layers, batch_size, self.hidden_size), device=device)


    def getModelName(self):
        return "GRU"