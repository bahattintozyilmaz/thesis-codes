import torch as t
import numpy as np
import torch.optim as optim

from pyt_sent2vec import Sent2Vec, masked_cross_entropy_loss
from data_loader import JsonS2VLoader

data_path = '/Users/baha/Personal/thesis/wikihowdumpall.clean.json'
task2vec_path = '/Users/baha/Personal/thesis/new-nn-models/task2vec-fixed-real'

embedding_size = 1000
word_embedding_size = 500
max_seq_len = 100
batch_size = 60
vocab_size = 10000
gradient_clip = 10.0
enable_cuda = False
num_epochs = 10
filter_cats = []

log_every = 50
save_every = 10000
save_backoff = 100

try:
    from overrides import *
except ImportError:
    pass

def log(*args, log_file=None):
    print(*args)
    if log_file:
        print(*args, file=log_file)
        log_file.flush()

def load_data():
    data_loader = JsonS2VLoader(data_path, num_words=vocab_size, longest_sent=max_seq_len, as_cuda=enable_cuda)
    if filter_cats:
        data_loader.filter(filter_cats)
    data_loader.load().preprocess()
    data_loader.filter_by_unknown_ratio(0.07)

    data_loader.word_converter.dump(task2vec_path+'.vocab.pkl')

    return data_loader

def create_model():
    model = Sent2Vec(encode_dim=embedding_size, embed_dim=word_embedding_size, embed_count=vocab_size, longest_seq=max_seq_len)
    return model

def train(model, data_loader):
    """
    trains given model
    """
    num_batches = (data_loader.get_total_triplets() + batch_size - 1) // batch_size

    if enable_cuda:
        model = model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters())

    best_loss = 1e8

    with open(task2vec_path+'.run.log', 'a') as logfile:
        for epoch in range(num_epochs):
            log('starting epoch ', epoch+1, log_file=logfile)
            total_loss = 0
            last_saved = -save_backoff
            for batchid, batch in enumerate(data_loader.get_triplets(batch_size=batch_size)):
                prv, cur, nxt = batch
                prv, prv_len = prv
                nxt, nxt_len = nxt
                prv_pred, nxt_pred = model(batch)
                optimizer.zero_grad()

                prv_loss = masked_cross_entropy_loss(prv_pred.contiguous(), t.autograd.Variable(prv), prv_len)
                nxt_loss = masked_cross_entropy_loss(nxt_pred.contiguous(), t.autograd.Variable(nxt), nxt_len)

                loss = prv_loss + nxt_loss
                loss.backward()

                t.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)

                optimizer.step()
                this_step_loss = loss.sum().data[0]
                total_loss += this_step_loss

                if batchid % log_every == 0:
                    log("\tBatch {}/{}, average loss: {}, current loss: {}".format(
                        batchid, num_batches, total_loss/(batchid+1), this_step_loss), log_file=logfile)

                if this_step_loss < best_loss and (last_saved+save_backoff) <= batchid:
                    log("\t\tSaving best at epoch {}, batch {}...".format(epoch, batchid), log_file=logfile)
                    t.save(model, task2vec_path+".best.pyt")
                    best_loss = this_step_loss
                    last_saved = batchid

                if batchid % save_every == 0:
                    log("\t\tSaving regularly at epoch {}, batch {}...".format(epoch, batchid), log_file=logfile)
                    t.save(model, task2vec_path+".regular.pyt")

            t.save(model, task2vec_path+".last_epoch.pyt".format(epoch))

    return model
