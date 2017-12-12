import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pickle

def V(tensor):
    return Variable(tensor, requires_grad=True)

def C(tensor):
    return Variable(tensor, requires_grad=False)

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def masked_cross_entropy_loss(logits, target, lengths):
    length = Variable(torch.LongTensor(lengths))

    if logits.is_cuda:
        length = length.cuda()

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = nn.functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def one_hot_encode(labels, max_val):
    size = list(labels.size()) + [max_val]
    labels_size = list(labels.size()) + [1]
    y_onehot = torch.FloatTensor(*size)
    if labels.is_cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()

    y_onehot.scatter_(len(size)-1, labels.view(*labels_size), 1)

    return y_onehot.contiguous()

class Sent2Vec(nn.Module):
    def __init__(self, encode_dim, embed_dim, embed_count, longest_seq):
        super(Sent2Vec, self).__init__()
        self.longest_sequence = longest_seq
        self.encode_dim = encode_dim

        self.embedding = nn.Embedding(num_embeddings=embed_count, embedding_dim=embed_dim)
        nn.init.uniform(self.embedding.weight, -0.1, 0.1)
        self.encoder = nn.GRU(input_size=embed_dim, hidden_size=encode_dim, batch_first=True)

        self.f_decoder = nn.GRU(input_size=embed_dim, hidden_size=encode_dim, batch_first=True)
        self.f_fc = nn.Linear(in_features=encode_dim, out_features=embed_count)
        nn.init.uniform(self.f_fc.weight, -0.1, 0.1)

        self.b_decoder = nn.GRU(input_size=embed_dim, hidden_size=encode_dim, batch_first=True)
        self.b_fc = nn.Linear(in_features=encode_dim, out_features=embed_count)
        nn.init.uniform(self.b_fc.weight, -0.1, 0.1)

    def forward(self, inp):
        cur, nxt, prv = inp[1], inp[2], inp[0]
        hidden = self.encode(cur)
        out = self.decode(hidden, nxt, prv)
        return out

    def encode(self, inp):
        inp, lengths = inp
        inp = C(inp)

        inits = C(inp.data.new(1, inp.size(0), self.encode_dim).zero_()).float()
        curr_embed = self.embedding(inp)

        out, imm = self.encoder(curr_embed, inits)
        padded_lengths = [i*inp.size(1)+v-1 for i, v in enumerate(lengths)]
        out_ = out.contiguous().view(-1, self.encode_dim)[padded_lengths, :]

        return out_

    def decode(self, imm, next_sent, prev_sent):
        def pad_embedding(emb):
            zeros = C(emb.data.new(emb.size(0), 1).zero_().long())
            return torch.cat([zeros, emb], 1)[:, :self.longest_sequence]
        next_sent, next_lengths = next_sent
        next_sent = C(next_sent)

        prev_sent, prev_lengths = prev_sent
        prev_sent = C(prev_sent)

        next_sent = pad_embedding(next_sent)
        prev_sent = pad_embedding(prev_sent)

        next_inits = C(next_sent.data.new(1, next_sent.size(0), self.encode_dim).zero_()).float()
        next_embed = self.embedding(next_sent)
        next_outs, _ = self.f_decoder(next_embed, next_inits)
        next_res = self.f_fc(next_outs)

        prev_inits = C(next_sent.data.new(1, prev_sent.size(0), self.encode_dim).zero_()).float()
        prev_embed = self.embedding(prev_sent)
        prev_outs, _ = self.b_decoder(prev_embed, prev_inits)
        prev_res = self.b_fc(prev_outs)

        return prev_res, next_res
