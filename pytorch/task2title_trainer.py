import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pyt_sent2vec import Sent2Vec, masked_cross_entropy_loss
from task2title import Task2Title
from data_loader import JsonS2VLoader
from utils import C

data_path = '/Users/baha/Personal/thesis/wikihowdumpall.clean.sent_processed.id_assigned.json'
task2vec_path = '/Users/baha/Personal/thesis/new-nn-models/task2vec-2018-01-31'
task2title_path = '/Users/baha/Personal/thesis/new-nn-models/task2title-2018-02-20'
task2vec_encoder_path = task2vec_path+'.cpu.pyt'

embedding_size = 1000
word_embedding_size = 300
max_seq_len = 100
batch_size = 128
vocab_size = 10000
gradient_clip = 20.0
enable_cuda = False
num_epochs = 10
task2title_hidden = 700
task2title_batch_size = 10
task2title_max_steps = 80
filter_cats = [] #['Finance and Business', 'Hobbies and Crafts', 'Home and Garden', 'Cars & Other Vehicles', 'Sports and Fitness', 'Pets and Animals', 'Work World', 'Youth', 'Philosophy and Religion', 'Health', 'Relationships', 'Education and Communications', 'Arts andEntertainment', 'Personal Care and Style', 'Family Life', 'Food and Entertaining']

log_every = 1
save_every = 1000
save_backoff = 100
use_cosine = True

try:
    from overrides import *
except ImportError:
    pass

def log(*args, log_file=None):
    print(*args)
    if log_file:
        print(*args, file=log_file)
        log_file.flush()

def create_model():
    global embedding_size, task2title_hidden
    model = Task2Title(input_size=embedding_size, inner_size=task2title_hidden)
    return model

def load_data():
    global data_path, vocab_size, max_seq_len, enable_cuda
    data_loader = JsonS2VLoader(data_path, num_words=vocab_size, longest_sent=max_seq_len, as_cuda=enable_cuda)
    if filter_cats:
        data_loader.filter(filter_cats)
    data_loader.load()
    data_loader.word_converter.load(task2vec_path+'.vocab.pkl')
    data_loader._prep_filter_empty_titles()
    data_loader._prep_split_sents()
    data_loader._prep_convert_sents()
    data_loader.filter_by_unknown_ratio(0.07)
    data_loader.filter_by_title_unknown_ratio(0.25);

    return data_loader

def load_vocab():
    pass

def load_encoder():
    global task2vec_encoder_path
    encoder = torch.load(task2vec_encoder_path).eval()
    return encoder

def encode(encoder, sents_tup):
    out = encoder.encode(sents_tup)[0].data
    return out

def prepare_batch(encoder, batch):
    global task2title_max_steps, embedding_size
    (sents_tup, seq_lens), res_tup = batch
    res = C(encode(encoder, res_tup))
    sents = encode(encoder, sents_tup)
    sents = sents.view(-1, task2title_max_steps, embedding_size)
    #norm = sents.norm(p=2, dim=2, keepdim=True)

    #return (sents.div(norm), seq_lens), res.div(res.norm(p=2, dim=1, keepdim=True))
    return (sents, seq_lens), res

def train(model, data_loader, encoder, training=None, testing=None):
    global task2title_path, task2title_batch_size
    num_batches = (data_loader.get_total_samples(source=training) + task2title_batch_size - 1) // task2title_batch_size

    if enable_cuda:
        model = model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters())

    best_loss = 1e8

    with open(task2title_path+'.run.log', 'a') as logfile, open(task2title_path+'.vectors.log', 'a') as vector_file:
        for epoch in range(num_epochs):
            log('starting epoch ', epoch+1, log_file=logfile)
            total_loss = 0
            last_saved = -save_backoff
            for batchid, batch in enumerate(data_loader.get_samples(batch_size=task2title_batch_size, max_seq=task2title_max_steps, source=training)):
                steps, results = prepare_batch(encoder, batch)

                predicted = model(steps)
                optimizer.zero_grad()

                length = len(batch[1][1])
                y = torch.FloatTensor(length).fill_(1)
                y = y.cuda() if enable_cuda else y

                loss = nn.functional.cosine_embedding_loss(predicted, results, C(y))

                loss.backward()

                torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)

                optimizer.step()
                this_step_loss = loss.sum().data[0]
                total_loss += this_step_loss

                if batchid % log_every == 0:
                    log("\tBatch {}/{}, average loss: {}, current loss: {}".format(
                        batchid, num_batches, total_loss/(batchid+1), this_step_loss), log_file=logfile)
                    #log("\t\tPred norms: ", (predicted).norm(dim=1).data.tolist(), log_file=logfile)
                    #log("\t\tGT norms: ", (results).norm(dim=1).data.tolist(), log_file=logfile)
                    log("\t\tDiff norms: ", (predicted-results).norm(dim=1).data.tolist(), log_file=logfile)

                if this_step_loss < best_loss and (last_saved+save_backoff) <= batchid:
                    log("\t\tSaving best at epoch {}, batch {}, loss {}...".format(epoch, batchid, this_step_loss), log_file=logfile)
                    torch.save(model, task2title_path+".best.pyt")
                    best_loss = this_step_loss
                    last_saved = batchid

                if batchid % save_every == 0:
                    log("\t\tSaving regularly at epoch {}, batch {}...".format(epoch, batchid), log_file=logfile)
                    torch.save(model, task2title_path+".regular.pyt")

                if epoch == num_epochs-1:
                    for v_id in range(length):
                        print('gt ', results.data[v_id,:].tolist(), file=vector_file)
                        print('pr ', predicted.data[v_id,:].tolist(), file=vector_file)

            # torch.save(model, task2title_path+".epoch-{}.pyt".format(epoch))

        if testing:
            model.eval()
            for batchid, batch in enumerate(data_loader.get_samples(batch_size=task2title_batch_size, max_seq=task2title_max_steps, source=testing)):
                steps, results = prepare_batch(encoder, batch)
                predicted = model(steps)

                length = len(batch[1][1])
                for v_id in range(length):
                        print('gt ', results.data[v_id,:].tolist(), file=vector_file)
                        print('pr ', predicted.data[v_id,:].tolist(), file=vector_file)

            model.train()
    return model

def cross_validate(data_loader, encoder):
    folds = 10
    for n in range(folds):
        # prepare training and testing 
        training = [d for (i,d) in enumerate(data_loader.data) if i%folds!=n]
        testing = [d for (i,d) in enumerate(data_loader.data) if i%folds==n]
        model = create_model()
        train(model, data_loader, encoder, training=training, testing=testing)

