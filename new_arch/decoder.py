import torch
from torch import nn
from torch.nn import functional as F

from dataset import MAX_PARTS, MAX_SENTS, EMBED_DIM, device

class Vec2Title(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Vec2Title, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        #print(output.grad_fn, hidden[0].grad_fn, hidden[1].grad_fn)
        return output, hidden
