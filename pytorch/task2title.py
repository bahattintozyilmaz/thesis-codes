import torch
from torch import nn
from utils import C, V, tensors

class Task2Title(nn.Module):
    def __init__(self, embed_dim, max_steps):
        super(Task2Title, self).__init__()
        self.inner_size = embed_dim
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=self.inner_size, num_layers=2,
                          batch_first=True)
        nn.init.uniform(self.gru.weight, -0.1, 0.1)
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        nn.init.uniform(self.linear.weight, -0.1, 0.1)

    def forward(self, inp):
        inp, lengths = inp
        inp = C(inp)
        prev_state = C(tensors.FloatTensor([1, inp.size(0), inp.size(2)]).zero_())
        out, imm = self.gru(inp, prev_state)

        # select only outputs at lengths[i]
        padded_lengths = [i*inp.size(1)+v-1 for i, v in enumerate(lengths)]
        out_ = out.contiguous().view(-1, self.inner_size)[padded_lengths, :]

        # then feed them to fully connected
        out_ = self.linear(out_)

        return out_
