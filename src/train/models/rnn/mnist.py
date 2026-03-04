import torch.nn as nn
import torch

class MNISTLSTM(nn.Module):

    def __init__(self, n_layers, hidden_size=256, num_classes=10):
        super().__init__()
        self.rnn = []
        input_size = 28
        for _ in range(n_layers):
            self.rnn.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1))
            input_size = hidden_size
        self.rnn = nn.ModuleList(self.rnn)
        self.linear = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.transpose(0, 1)
        for rnn in self.rnn:
            x, _ = rnn(x) # [seq, batch, hidden]
        x = x.transpose(0, 1) # [batch, seq, hidden]
        x = x[:, -1]
        x = self.linear(x)
        return x
    
    
    

class MNISTGRU(nn.Module):

    def __init__(self, n_layers, hidden_size=256, num_classes=10):
        super().__init__()
        self.rnn = []
        input_size = 28
        for _ in range(n_layers):
            self.rnn.append(nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1))
            input_size = hidden_size
        self.rnn = nn.ModuleList(self.rnn)
        self.linear = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.transpose(0, 1)
        for rnn in self.rnn:
            x, _ = rnn(x) # [seq, batch, hidden]
        x = x.transpose(0, 1) # [batch, seq, hidden]
        x = x[:, -1]
        x = self.linear(x)
        return x
    
    
# @register_model
def mnist_lstm_128x1(*args, **kwargs):
    return MNISTLSTM(n_layers=1, hidden_size=128)

def mnist_lstm_128x2(*args, **kwargs):
    return MNISTLSTM(n_layers=2, hidden_size=128)

def mnist_lstm_64x1(*args, **kwargs):
    return MNISTLSTM(n_layers=1, hidden_size=64)

def mnist_gru_128x1(*args, **kwargs):
    return MNISTGRU(n_layers=1, hidden_size=128)

def mnist_gru_64x1(*args, **kwargs):
    return MNISTGRU(n_layers=1, hidden_size=64)

def mnist_gru_128x2(*args, **kwargs):
    return MNISTGRU(n_layers=2, hidden_size=128)


if __name__ == "__main__":
    model = mnist_lstm_128x1()
    x = torch.randn(7, 1, 28, 28)
    y = model(x)
    print(model)
    print(y.shape)
