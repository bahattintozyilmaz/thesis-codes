import nltk
import json
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordTokenizer
import pickle
import torch
import multiprocessing
from os.path import exists as file_exists
import numpy as np
from sent2vec import Sent2vecModel

word_tokenize = TreebankWordTokenizer().tokenize
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

MAX_PARTS = 10
MAX_SENTS = 8
MAX_WORDS = None
MAX_TITLE_WORDS = None
EMBED_DIM = 600

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


s2v = None

def encode_sents(sents, pad_to):
    global s2v

    if not s2v:
        s2v = Sent2vecModel()
        s2v.load_model('/Users/baha/personal/thesis-codes/wiki_unigrams.bin')

    sents = [' '.join(word_tokenize(s)).lower() for s in sents]

    res = s2v.embed_sentences(sents)
    
    if res.shape[0] < pad_to:
        res = np.pad(res, ((0, pad_to-res.shape[0]),(0, 0)), 'constant')

    return res


class WordConverter():
    EOS = 0
    SOS = 1
    UNK = 2

    def __init__(self, num_words):
        self.num_words = num_words
        self.word_counts = defaultdict(int)
        self.word_sentences = defaultdict(int)
        self.word_dict = None
        self.word_idict = None

        self.translations = {"''": '"', "``": '"'}

    def feed_words(self, words, word_set=None):
        if not isinstance(words, list):
            words = [words]

        for w in words:
            ww = w.lower().strip()
            ww = self.translations.get(ww, ww)
            self.word_counts[ww] += 1
            if word_set is not None and (ww not in word_set):
                self.word_sentences[ww] += 1
                word_set.add(ww)

    def feed_sentence(self, sent, word_set=None):
        words = word_tokenize(sent)
        self.feed_words(words, word_set)

    def feed_sample(self, sents):
        word_set = set()
        for sent in sents:
            self.feed_sentence(sent, word_set)

    def feed_other(self, other):
        other_word_counts, other_word_sentences = other

        for k, v in other_word_counts.items():
            self.word_counts[k] += v

        for k, v in other_word_sentences.items():
            self.word_sentences[k] += v

    def finalize(self):
        import math
        if not (self.word_dict and self.word_idict):
            num_words = self.num_words
            if num_words is None:
                num_words = len(self.word_counts) + 3
            count_word_list = [(self.word_sentences[w], c, w) for w, c in self.word_counts.items()]
            count_word_list.sort(reverse=True)
            count_word_list = count_word_list[:(num_words-3)]

            count_word_list = [(0, 0, '<eos>'), (0, 0, '<sos>'), (0, 0, '<unk>')] + count_word_list[:(num_words-3)]
            self.word_dict = dict((e[-1], i) for i, e in enumerate(count_word_list))
            self.word_idict = dict((i, e[-1]) for i, e in enumerate(count_word_list))

        return self.word_dict, self.word_idict

    def translate(self, sent):
        words = [w.lower().strip() for w in word_tokenize(sent)]
        translated = [self.word_dict.get(w, self.word_dict['<unk>']) for w in words] + [self.word_dict['<eos>']]
        return translated
        
    def recreate(self, list_):
        return ' '.join(self.word_idict.get(i, '<nid>') for i in list_)

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.word_dict, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            other = cls(0)
            other.word_dict = pickle.load(f)
            other.word_idict = dict((v,k) for k,v in other.word_dict.items())

            return other


class Dataset:
    def clear(self):
        data = []

        def clear_parts(parts):
            return [p for p in parts if p.get('steps')]

        for d in self.data:
            d['parts'] = clear_parts(d['parts'])

            if d['parts']:
                data.append(d)

        self.data = data

    @classmethod
    def from_file(cls, fn):
        with open(fn) as f:
            data = json.load(f)
            return cls.from_data(data)

    @classmethod
    def from_data(cls, data, title_converter=None):
        other = cls()
        other.data = data
        other.clear()

        if title_converter:
            other.title_converter = title_converter

        return other

    @classmethod
    def load(cls, fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)
        #if file_exists(fn+'.title_data'):
        #    self.title_converter = WordConverter.load(fn+'.title_data')
    
    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump(self, fn)
        #if isinstance(self.title_converter, WordConverter):
        #    self.title_converter.dump(fn+'.title_data')

    def split(self, ways):
        """ returns 2 Dataset's, one has (ways-1)/ways and one has 1/ways
        """

        train = Dataset()
        train.title_converter = getattr(self, 'title_converter', None)
        train.data = [d for (i, d) in enumerate(self.data) if i%ways != 0]

        test = Dataset()
        test.title_converter = getattr(self, 'title_converter', None)
        test.data = [d for (i, d) in enumerate(self.data) if i%ways == 0]

        return train, test

    def process_title_words(self, cap_at=None):
        # ~%95 coverage is at 7500 words
        self.title_converter = WordConverter(cap_at)
        for sample in self.data:
            self.title_converter.feed_sentence(sample['title'])
        self.title_converter.finalize()

    def create_part_matrix(self, part):
        return encode_sents(part['steps'][0:MAX_SENTS], MAX_SENTS)
    
    def create_sample_data(self, sample):
        parts = []
        for p in sample['parts'][0:MAX_PARTS]:
            parts.append(self.create_part_matrix(p))
        
        for i in range(len(parts), MAX_PARTS):
            parts.append(np.zeros_like(parts[0]))

        parts = torch.tensor(np.array(parts), dtype=torch.float, device=device).unsqueeze(0)

        title = torch.tensor(self.title_converter.translate(sample['title']), dtype=torch.long, device=device)
        title = title.view(title.size(0), 1)

        return (parts, title) 

    def get_samples(self, data=None):
        data = data or self.data

        for sample in sorted(data, key=lambda e: len(e['title'])):
            if sample['parts']:
                yield self.create_sample_data(sample)