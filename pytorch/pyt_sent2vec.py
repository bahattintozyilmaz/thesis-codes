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

def masked_cross_entropy_loss(logits, target, lengths, weight):
    length = Variable(torch.LongTensor(lengths))

    if logits.is_cuda:
        length = length.cuda()
        weight = weight.cuda()

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = nn.functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    #print(target_flat)
    weight_flat = torch.index_select(weight, dim=0, index=target_flat.squeeze())
    # losses: (batch, max_len)
    #print('lf', losses_flat)
    #print('wf', weight_flat)
    losses_flat = losses_flat.mul(weight_flat.unsqueeze(1))
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum(dim=1).div(length.float())
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
        self.encoder = nn.GRU(input_size=embed_dim, hidden_size=encode_dim, batch_first=True)

        self.f_decoder = nn.GRU(input_size=embed_dim, hidden_size=encode_dim, batch_first=True)
        self.b_decoder = nn.GRU(input_size=embed_dim, hidden_size=encode_dim, batch_first=True)

        self.fc = nn.Linear(in_features=encode_dim, out_features=embed_count)

        self.init_weights()

    def init_weights(self):
        nn.init.uniform(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform(self.fc.weight, -0.1, 0.1)

    def apply_word2vec_weights(self, word2vec, embedding_keys):
        n = 0
        not_found = []
        translation = {'<eos>': '</s>'}
        for i, word in enumerate(embedding_keys):
            word = translation.get(word, word)
            if word in word2vec:
                self.embedding.weight.data[i,:] = torch.FloatTensor(word2vec[word])
                n+=1
            else:
                not_found.append(word)
        print('applied ', n, ' word2vecs')
        print(not_found)

    def forward(self, inp):
        cur, nxt, prv = inp[1], inp[2], inp[0]
        hidden, _ = self.encode(cur)
        out = self.decode(hidden, nxt, prv)
        return out

    def encode(self, inp):
        inp, lengths = inp
        inp = C(inp)

        curr_embed = self.embedding(inp)

        out, imm = self.encoder(curr_embed)
        out_ = imm.squeeze(0)

        return nn.functional.normalize(out_, p=2, dim=1), out.div(out.norm(p=2, dim=2, keepdim=True))

    def decode(self, hidden, next_sent, prev_sent):
        def pad_embedding(emb):
            zeros = C(emb.data.new(emb.size(0), 1).zero_().long())
            return torch.cat([zeros, emb], 1)[:, :self.longest_sequence]
        next_sent, next_lengths = next_sent
        next_sent = C(next_sent)

        prev_sent, prev_lengths = prev_sent
        prev_sent = C(prev_sent)

        next_sent = pad_embedding(next_sent)
        prev_sent = pad_embedding(prev_sent)

        inits = hidden.view(1, hidden.size(0), hidden.size(1))
        
        next_embed = self.embedding(next_sent)
        next_outs, _ = self.f_decoder(next_embed, inits)
        next_res = self.fc(next_outs)

        prev_embed = self.embedding(prev_sent)
        prev_outs, _ = self.b_decoder(prev_embed, inits)
        prev_res = self.fc(prev_outs)

        return prev_res, next_res
