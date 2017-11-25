import nltk
import json
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordTokenizer
import pickle
import torch
import multiprocessing

word_tokenize = TreebankWordTokenizer().tokenize
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def _process_sample(sample):
    """
    sample: {steps: [str]}
    on return, sample['steps'] is a list of sentences
    """
    arr = []
    translations = {'``': '"', "''": '"'}
    for s in sample['steps']:
        s = s.replace('``', '"').replace('\'\'', '"').replace('``', '"')
        arr.extend([translations.get(s_, s_) for s_ in [s.strip() for s in sent_tokenizer.tokenize(s)]])
    sample['steps'] = [s.lower() for s in arr if s]
    return sample

def _process_sents(samples):
    wc = WordConverter(num_words=-1)
    for sample in samples:
        for s in sample['steps']:
            wc.feed_sentence(s)
        
    return wc.word_counts

def _process_translate(wc, sample):
    sample['processed'] = [wc.translate(s) for s in sample['steps']]
    return sample

class WordConverter():
    def __init__(self, num_words):
        self.num_words = num_words
        self.word_counts = defaultdict(int)
        self.word_dict = None
        self.word_idict = None

    def feed_words(self, words):
        if not isinstance(words, list):
            words = [words]

        for w in words:
            self.word_counts[w.lower().strip()] += 1

    def feed_sentence(self, sent):
        words = word_tokenize(sent)
        self.feed_words(words)

    def feed_dict(self, dict_):
        for k, v in dict_.items():
            self.word_counts[k] += v

    def finalize(self):
        if not (self.word_dict and self.word_idict):
            count_word_list = [(c, w) for w, c in self.word_counts.items()]
            count_word_list.sort(reverse=True)
            count_word_list = [(0, '<eos>'), (0, '<unk>')] +count_word_list[:(self.num_words-2)]
            self.word_dict = dict((e[1], i) for i, e in enumerate(count_word_list))
            self.word_idict = dict((i, e[1]) for i, e in enumerate(count_word_list))

        return self.word_dict, self.word_idict

    def translate(self, sent):
        words = [w.lower().strip() for w in word_tokenize(sent)]
        translated = [self.word_dict.get(w, self.word_dict['<unk>']) for w in words]
        return translated
        
    def recreate(self, list_):
        return ' '.join(self.word_idict.get(i, '<nid>') for i in list_)

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.word_dict, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            pickle.load(self.word_dict, f)
        self.word_idict = dict((v,k) for k,v in self.word_dict.items())

class JsonS2VLoader():
    def __init__(self, filename, num_words=10000, longest_sent=100, as_cuda=False):
        self.filename = filename
        self.num_words = num_words
        self.longest_sent = longest_sent
        self.as_cuda = as_cuda
        self.filter_func = lambda x: True
        self.data = None
        self.word_converter = WordConverter(num_words=num_words)
        self._cached_num_total_triplets = None
        self.num_cpus = multiprocessing.cpu_count()

    def filter(self, cats):
        """ creates filteration function and
        """
        cat_filter_generator = lambda cts: (lambda x: any((cat in x.get('cat', [])) for cat in cts))
        self.filter_func = cat_filter_generator(cats)

        return self

    def load(self):
        with open(self.filename, 'r') as f:
            self.data = json.load(f)

        if self.filter_func:
            self.data = [s for s in self.data if self.filter_func(s)]

        return self

    def preprocess(self):
        pool = multiprocessing.Pool(processes=self.num_cpus)
        self.data = pool.map(_process_sample, self.data)
        pool.terminate()
        print('Done splitting sentences')

        pool = multiprocessing.Pool(processes=self.num_cpus)
        all_counts = pool.map(_process_sents, (self.data[i::self.num_cpus] for i in range(self.num_cpus)))
        pool.terminate()
        for count in all_counts:
            self.word_converter.feed_dict(count)

        print('Done feeding word counter')
        self.word_converter.finalize()
        print('Done creating word ids')
        # pre calculate everything, why not!
        pool = multiprocessing.Pool(processes=self.num_cpus)
        self.data = pool.starmap(_process_translate, ((self.word_converter, s) for s in self.data))
        pool.terminate()
        print('Done precalculating sentence vectors')

        pool = multiprocessing.Pool(processes=self.num_cpus)
        all_counts = pool.map(_process_sents, (self.data[i::self.num_cpus] for i in range(self.num_cpus)))
        pool.terminate()
        for count in all_counts:
            self.word_converter.feed_dict(count)

        print('Done creating groups')

        return self

    def _pack_sent_tensors(self, sents):
        lens = [min(len(s), self.longest_sent) for s in sents]
        padded = torch.LongTensor([(s + ([0]*self.longest_sent))[:self.longest_sent] for s in sents])
        if self.as_cuda:
            padded = padded.cuda()
        return padded, lens

    def get_total_triplets(self):
        if not self._cached_num_total_triplets:
            s = sum((len(s['steps'])-2 for s in self.data), 0)
            self._cached_num_total_triplets = s

        return self._cached_num_total_triplets

    def get_triplets(self, batch_size=60):
        prv, cur, nxt = [], [], []
        for s in self.data:
            for i in range(len(s['steps'])-2):
                prv.append(s['processed'][i])
                cur.append(s['processed'][i+1])
                nxt.append(s['processed'][i+2])
                if len(cur) >= batch_size:
                    yield (self._pack_sent_tensors(prv), self._pack_sent_tensors(cur), self._pack_sent_tensors(nxt))
                    prv, cur, nxt = [], [], []

        if cur:
            yield (self._pack_sent_tensors(prv), self._pack_sent_tensors(cur), self._pack_sent_tensors(nxt))
