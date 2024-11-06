import torch
from torch import nn

import numpy as np

class Model(nn.Module):

    def __init__(self, input_size, size, rnn_type, target_size, n_blocks, device='cpu'):
        super(Model, self).__init__()
        self.embed = nn.Linear(input_size, size, device=device)
        self.output = nn.Linear(size, target_size, device=device)
        self.rnn_type = rnn_type
        self.size = size
        self.device = device
        self.n_blocks = n_blocks
        if rnn_type == 'LSTM':
            self.rnn = [nn.LSTM(input_size=size, hidden_size=size, device=device) for _ in range(n_blocks)]
        if rnn_type == 'GRU':
            self.rnn = [nn.GRU(input_size=size, hidden_size=size, device=device) for _ in range(n_blocks)]
        if rnn_type == 'RNN':
            self.rnn = [nn.RNN(input_size=size, hidden_size=size, device=device) for _ in range(n_blocks)]


    def forward(self, x):
        x = torch.relu(self.embed(x))
        if self.rnn_type == 'LSTM':
            h = torch.zeros([1, 1, self.size], device=self.device)
            c = torch.zeros([1, 1, self.size], device=self.device)
            for i in range(self.n_blocks):
                x, (h, c) = self.rnn[i](x, (h, c))
        if self.rnn_type == 'GRU':
            h = torch.zeros([1, 1, self.size], device=self.device)
            for i in range(self.n_blocks):
                x, h = self.rnn[i](x, h)
        if self.rnn_type == 'RNN':
            h = torch.zeros([1, 1, self.size], device=self.device)
            for i in range(self.n_blocks):
                x, h = self.rnn[i](x, h)
        y = torch.sigmoid(self.output(x))
        return y


def get_model(config):
    return Model(config['input'], config['hidden'], config['model'], config['targets'], config['n_blocks'], config['device'])


def pd_to_tensor(x, y, device='cpu'):
    xx = x.to_numpy(dtype=np.float32)
    yy = y.to_numpy(dtype=np.float32)
    xx = torch.tensor(xx, device=device).unsqueeze(1)
    yy = torch.tensor(yy, device=device).unsqueeze(1)
    return xx, yy
