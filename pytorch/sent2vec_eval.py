import torch as t
import numpy as np
import torch.optim as optim

from pyt_sent2vec import Sent2Vec, masked_cross_entropy_loss
from data_loader import JsonS2VLoader
from dataset_utils import load_word2vec

data_path = '/Users/baha/Personal/thesis/wikihowdumpall.clean.json'
word2vec_path = '/Users/baha/Personal/thesis/vectors.txt'
task2vec_path = '/Users/baha/Personal/thesis/new-nn-models/task2vec-take-six'
model_path = task2vec_path+'.final.pyt.cpu'

embedding_size = 1000
word_embedding_size = 300
max_seq_len = 100
batch_size = 60
vocab_size = 10000
gradient_clip = 10.0
enable_cuda = False
num_epochs = 10
filter_cats = []

log_every = 10
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

def load_data(data_path=data_path):
    data_loader = JsonS2VLoader(data_path, num_words=vocab_size, longest_sent=max_seq_len, as_cuda=enable_cuda)
    if filter_cats:
        data_loader.filter(filter_cats)
    data_loader.load()
    data_loader.word_converter.load(task2vec_path+'.vocab.pkl')
    data_loader._prep_filter_empty_titles()
    data_loader._prep_split_sents()
    data_loader._prep_convert_sents()
    data_loader.filter_by_unknown_ratio(0.07)

    return data_loader

def load_model(path=model_path):
    model = t.load(path).eval()
    return model

def sample(data, pred, gt, lens, i, suffix, logfile):
    i = max(i, len(lens)-1)
    pred = pred[i,:,:]
    gt = gt[i,:]
    _, pred_indices = pred.max(dim=1)
    length = lens[i]
    log(suffix, 'gt:   ', data.word_converter.recreate(gt.tolist()[:length]), log_file=logfile)
    log(suffix, 'pred: ', data.word_converter.recreate(pred_indices.tolist()[:length+3]), log_file=logfile)
    
def eval(model, data_loader):
    """
    evals given model
    """
    num_batches = (data_loader.get_total_triplets() + batch_size - 1) // batch_size
    #class_weights = t.autograd.Variable(t.FloatTensor(data_loader.word_converter.tfidfs))
    if enable_cuda:
        model = model.cuda()
        #class_weights = class_weights.cuda()

    class_weights = t.autograd.Variable(t.ones(vocab_size))

    best_loss = 1e8

    with open(task2vec_path+'.eval.log', 'a') as logfile:
        for epoch in range(num_epochs):
            log('starting epoch ', epoch+1, log_file=logfile)
            total_loss = 0
            last_saved = -save_backoff
            for batchid, batch in enumerate(data_loader.get_triplets(batch_size=batch_size)):
                prv, cur, nxt = batch
                prv, prv_len = prv
                nxt, nxt_len = nxt
                prv_pred, nxt_pred = model(batch)

                prv_loss = masked_cross_entropy_loss(prv_pred.contiguous(), t.autograd.Variable(prv), prv_len, class_weights)
                nxt_loss = masked_cross_entropy_loss(nxt_pred.contiguous(), t.autograd.Variable(nxt), nxt_len, class_weights)

                loss = prv_loss + nxt_loss
                
                this_step_loss = loss.mean().data[0]
                total_loss += this_step_loss

                log('\tBatch', batchid, prv_loss.mean().data[0], nxt_loss.mean().data[0], log_file=logfile)
                sample(data_loader, prv_pred.data, prv, prv_len, 0, '\t\tbw', logfile)
                sample(data_loader, nxt_pred.data, nxt, nxt_len, 0, '\t\tfw', logfile)

                if batchid % log_every == 0:
                    log("\tBatch {}/{}, average loss: {}, current loss: {}".format(
                        batchid, num_batches, total_loss/(batchid+1), this_step_loss), log_file=logfile)

    return model
