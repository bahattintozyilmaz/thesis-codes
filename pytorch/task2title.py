import torch
from torch import nn
from utils import C, V, tensors

class Task2Title(nn.Module):
    def __init__(self, input_size, inner_size):
        super(Task2Title, self).__init__()
        self.inner_size = inner_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=self.inner_size, num_layers=2,
                          batch_first=True)
        nn.init.uniform(self.gru.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform(self.gru.weight_ih_l0, -0.1, 0.1)
        self.linear = nn.Linear(in_features=self.inner_size, out_features=input_size)
        nn.init.uniform(self.linear.weight, -0.1, 0.1)

    def forward(self, inp):
        inp, lengths = inp
        inp = C(inp)
        out, imm = self.gru(inp)

        # select only outputs at lengths[i]
        padded_lengths = [i*inp.size(1)+v-1 for i, v in enumerate(lengths)]
        print(padded_lengths)
        out_ = out.contiguous().view(-1, self.inner_size)[padded_lengths, :]

        # then feed them to fully connected
        out_ = self.linear(out_)

        return out_
