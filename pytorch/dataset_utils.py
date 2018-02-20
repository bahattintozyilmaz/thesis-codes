import json
import multiprocessing
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from data_loader import WordConverter, sent_tokenizer
from pyt_sent2vec import Sent2Vec

def load_vocab(filename):
    word_converter = WordConverter(num_words=10000)
    word_converter.load(filename)
    return word_converter

def load_dataset(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_model(filename):
    return torch.load(filename)

def add_name(filename, part):
    basename, ext = os.path.splitext(filename)
    return basename + '.' + part + ext

def _split_sentences_func(sample):
    """
    sample: {steps: [str]}
    on return, sample['steps'] is a list of sentences
    """
    arr = []
    translations = {'``': '"', "''": '"'}
    for s in sample['steps']:
        s = s.replace('``', '"').replace('\'\'', '"').replace('``', '"').replace('.', ' . ').replace('-', ' - ').replace('/', ' / ')
        arr.extend([translations.get(s_, s_) for s_ in [s.strip() for s in sent_tokenizer.tokenize(s)]])
    sample['steps'] = [s.lower() for s in arr if s]
    return sample

def split_dataset_sentences(filename, outname=None):
    if not outname:
        outname = add_name(filename, 'sent_processed')

    data = load_dataset(filename)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    data = pool.map(_split_sentences_func, data)
    pool.terminate()

    with open(outname, 'w') as f:
        json.dump(data, f)

    return data

def _encode(model, wc, sents, padded_length=100):
    single_sent = not isinstance(sents, list)
    
    if single_sent:
        sents = [sents]

    pad_sent = lambda vec: (vec + ([0]*padded_length))[:padded_length]

    encoded_sents = [wc.translate(sent) for sent in sents]
    padded_sents = [pad_sent(sent) for sent in encoded_sents]
    inp = (torch.LongTensor(padded_sents), [min(len(sent), padded_length) for sent in encoded_sents])
    out = model.encode(inp).data
    
    if single_sent:
        out = out[0, :]
    
    return out

def convert_dataset_to_vec(model, wc, filename, outname=None):
    if not outname:
        outname = add_name(filename, 'encoded_dict')

    data = load_dataset(filename)
    new_data = {}

    for i in range(len(data)):
        if not data[i]['steps'] or not data[i]['cat']:
            continue
        steps_vec = _encode(model, wc, data[i]['steps'])
        title_vec = _encode(model, wc, data[i]['title'])
        
        new_data[data[i]['title']] = steps_vec.numpy()
        for j, s in enumerate(data[i]['steps']):
            new_data[s] = steps_vec[j, :].numpy()

        if i % 100 == 0:
            print(i, '/', len(data))

    with open(outname, 'w') as f:
        pickle.dump(new_data, f)

    return data

def _tok(text,ts=False):
    
    '''
    Usage: tokenized_text = tok(text,token_list)
    If token list is not provided default one will be used instead.
    '''

    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"',"'ve", "'d", "'m", "n't"]

    for t in ts:
        text = text.replace(t,' ' + t + ' ')
    return text

def prep_word2vec_input(filename, outname):
    data = load_dataset(filename)

    with open(outname, 'w') as f:
        for sample in data:
            for sent in sample['steps']:
                f.write(_tok(sent)+'\n')

def load_word2vec(path):
    with open(path) as f:
        # data = ['word v1 v2 v3 ...', ...]
        data = f.read().split('\n')[1:]

    # data = [['word', 'v1', 'v2', ...], ...]
    data = [s.rstrip().split(' ') for s in data]
    # data = {'word', [v1, v2, ...], ...}
    data = {s[0]: [float(n) for n in s[1:]] for s in data}

    return data

def save_generic(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_generic(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_vectors(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    pr = [eval(l[3:]) for l in lines if l.startswith('pr ')]
    gt = [eval(l[3:]) for l in lines if l.startswith('gt ')]

    return np.array(pr), np.array(gt)

def nearest_ind(vec, neighbors):
    dists = neighbors - vec
    dists = (dists*dists).sum(axis=1)
    return dists.argmin()

def evaluate(pr, gt):
    correct = 0
    for i in tqdm(range(pr.shape[0])):
        nearest = nearest_ind(pr[i,:], gt)
        if i==nearest:
            correct += 1

    return correct, pr.shape[0]
