import torch as t
import numpy as np
import torch.optim as optim

from pyt_sent2vec import Sent2Vec, masked_cross_entropy_loss, one_hot_encode 
from data_loader import JsonS2VLoader

data_path = '/Users/baha/Personal/thesis/wikihowdumpall.clean.json'
out_path = '/Users/baha/Personal/thesis/nn-models/task2vec'

embedding_size = 1000
word_embedding_size = 500
max_seq_len = 100
batch_size = 60
vocab_size = 10000
log_every = 100
gradient_clip = 5.0
enable_cuda = False

num_epochs = 10

try:
    from overrides import *
except ImportError:
    pass

def log(*args, log_file=None):
    print(*args)
    if log_file:
        print(*args, file=log_file)
        log_file.flush()

def train(log_every=100):
    data_loader = JsonS2VLoader(data_path, num_words=vocab_size, longest_sent=max_seq_len, as_cuda=enable_cuda)
    data_loader.load().preprocess()

    data_loader.word_converter.dump(out_path+'.vocab.pkl')
    num_batches = (data_loader.get_total_triplets() + batch_size - 1) // batch_size

    sent2vec = Sent2Vec(encode_dim=embedding_size, embed_dim=word_embedding_size, embed_count=vocab_size, longest_seq=max_seq_len)
    if enable_cuda:
        sent2vec = sent2vec.cuda()
    sent2vec.train()

    optimizer = optim.Adam(sent2vec.parameters())

    best_loss = 1e8

    with open(out_path+'.run.log', 'a') as logfile:
        for epoch in range(num_epochs):
            log('starting epoch ', epoch+1, log_file=logfile)
            total_loss = 0
            for batchid, batch in enumerate(data_loader.get_triplets()):
                prv, cur, nxt = batch
                prv, prv_len = prv
                nxt, nxt_len = nxt
                prv_pred, nxt_pred = sent2vec(batch)

                prv_loss = masked_cross_entropy_loss(prv_pred.contiguous(), t.autograd.Variable(prv), prv_len)
                nxt_loss = masked_cross_entropy_loss(nxt_pred.contiguous(), t.autograd.Variable(nxt), nxt_len)

                loss = prv_loss + nxt_loss
                loss.backward()

                t.nn.utils.clip_grad_norm(sent2vec.parameters(), gradient_clip)

                optimizer.step()
                this_step_loss = loss.sum().data[0]
                total_loss += this_step_loss
                
                if batchid % log_every == 0:
                    log("\tBatch {}/{}, average loss: {}, current loss: {}".format(
                        batchid, data_loader.get_total_triplets(), total_loss/(batchid+1), this_step_loss), log_file=logfile)

                if this_step_loss < best_loss:
                    log("\t\tSaving best at epoch {}, batch {}...".format(epoch, batchid), log_file=logfile)
                    t.save(sent2vec, out_path+".best.pyt")
                    best_loss = this_step_loss

            t.save(sent2vec, out_path+".epoch-{}.pyt".format(epoch))

    return sent2vec