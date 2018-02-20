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
        s = s.replace('``', '"').replace('\'\'', '"').replace('``', '"').replace('.', ' . ').replace('-', ' - ').replace('/', ' / ')
        arr.extend([translations.get(s_, s_) for s_ in [s.strip() for s in sent_tokenizer.tokenize(s)]])
    sample['steps'] = [s.lower() for s in arr if s]
    return sample

def _process_sents(samples):
    wc = WordConverter(num_words=-1)
    for sample in samples:
        wc.feed_sample(sample['steps'])

    return wc.word_counts, wc.word_sentences, wc.total_docs, wc.total_words

def _process_translate(wc, sample):
    sample['processed'] = [wc.translate(s) for s in sample['steps']]
    sample['title_proc'] = wc.translate(sample['title'])
    return sample

class WordConverter():
    def __init__(self, num_words):
        self.num_words = num_words
        self.total_words = 0
        self.total_docs = 0
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
            self.total_words += 1
            if word_set is not None and (ww not in word_set):
                self.word_sentences[ww] += 1
                word_set.add(ww)

    def feed_sentence(self, sent, word_set=None):
        words = word_tokenize(sent)
        self.feed_words(words, word_set)

    def feed_sample(self, sents):
        word_set = set()
        self.total_docs += 1
        for sent in sents:
            self.feed_sentence(sent, word_set)

    def feed_other(self, other):
        other_word_counts, other_word_sentences, other_total_docs, other_total_words = other

        for k, v in other_word_counts.items():
            self.word_counts[k] += v

        for k, v in other_word_sentences.items():
            self.word_sentences[k] += v

        self.total_words += other_total_words
        self.total_docs += other_total_docs

    def finalize(self):
        import math
        if not (self.word_dict and self.word_idict):
            count_word_list = [(c, w) for w, c in self.word_counts.items()]
            count_word_list.sort(reverse=True)
            count_word_list = count_word_list[:(self.num_words-2)]
            mean_count = sum([c for (c,w) in count_word_list])/len(count_word_list)
            tf_idfs = [(c/mean_count)*math.log((1+self.total_docs)/(1+self.word_sentences[w])) 
                       for (c, w) in count_word_list]

            count_word_list = [(0, '<eos>'), (0, '<unk>')] + count_word_list[:(self.num_words-2)]
            self.word_dict = dict((e[1], i) for i, e in enumerate(count_word_list))
            self.word_idict = dict((i, e[1]) for i, e in enumerate(count_word_list))
            self.ordered_words = [c[1] for c in count_word_list]

            mean_tf_idfs = sum(tf_idfs)/len(tf_idfs)

            self.tfidfs = [1, mean_tf_idfs] + tf_idfs

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
            pickle.dump(self.tfidfs, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.word_dict = pickle.load(f)
            try:
                self.tfidfs = pickle.load(f)
            except:
                pass
        self.word_idict = dict((v,k) for k,v in self.word_dict.items())
        self.ordered_words = [c[1] for c in sorted(list(self.word_idict.items()))]

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

        self.data = [s for s in self.data if s.get('status') != 'rejected']

        if self.filter_func:
            self.data = [s for s in self.data if self.filter_func(s)]

        del self.filter_func

        return self

    def _prep_split_sents(self):
        pool = multiprocessing.Pool(processes=self.num_cpus)
        self.data = pool.map(_process_sample, self.data)
        pool.terminate()
        print('Done splitting sentences')
        return self

    def _prep_feed_word_counter(self):
        pool = multiprocessing.Pool(processes=self.num_cpus)
        all_counts = pool.map(_process_sents, (self.data[i::self.num_cpus] for i in range(self.num_cpus)))
        pool.terminate()
        for other in all_counts:
            self.word_converter.feed_other(other)

        print('Done feeding word counter')
        self.word_converter.finalize()
        print('Done creating word ids')
        return self

    def _prep_convert_sents(self):
        # pre calculate everything, why not!
        pool = multiprocessing.Pool(processes=self.num_cpus)
        self.data = pool.starmap(_process_translate, ((self.word_converter, s) for s in self.data))
        pool.terminate()
        print('Done precalculating sentence vectors')
        return self

    def _prep_filter_empty_titles(self):
        self.data = [s for s in self.data if (s['steps'] and s['cat'])]
        return self

    def preprocess(self):
        self._prep_split_sents()
        self._prep_feed_word_counter()
        self._prep_convert_sents()
        self._prep_filter_empty_titles()
   
        return self

    def filter_by_unknown_ratio(self, ratio):
        def unknown_ratio(s):
            totes = list(zip(*[(proc.count(1), len(proc)) for proc in s['processed']]))
            return sum(totes[0])/sum(totes[1])
        self.data = [s for s in self.data if unknown_ratio(s)<ratio]

    def filter_by_title_unknown_ratio(self, ratio):
        def unknown_ratio_tit(s):
            return s['title_proc'].count(1)/len(s['title_proc'])
        self.data = [s for s in self.data if unknown_ratio_tit(s)<ratio]

    def split_training_validation_test(self, random_seed, ratio=0.05):
        import random
        cut = round(ratio * len(self.data))
        random.seed(random_seed)
        random.shuffle(self.data)
        self.data, self.val, self.test = self.data[:-2*cut], self.data[-2*cut:-cut], self.data[-cut:]
        
        return self

    def _pack_sent_tensors(self, sents):
        lens = [min(len(s), self.longest_sent) for s in sents]
        padded = torch.LongTensor([(s + ([0]*self.longest_sent))[:self.longest_sent] for s in sents])
        if self.as_cuda:
            padded = padded.cuda()
        return padded, lens

    def _pack_sample_tensors(self, samples, max_seq):
        orig_lens = [len(s) for s in samples]
        sents = []

        for s in samples:
            s_ = (s + ([[0]] * max_seq))[:max_seq]
            sents.extend(s_)

        return self._pack_sent_tensors(sents), orig_lens

    def get_total_triplets(self):
        s = sum((len(s['steps'])-2 for s in self.data), 0)
        return s

    def get_triplets(self, batch_size=60, source=None):
        if not source:
            source = self.data

        prv, cur, nxt = [], [], []
        for s in source:
            for i in range(len(s['steps'])-2):
                prv.append(s['processed'][i])
                cur.append(s['processed'][i+1])
                nxt.append(s['processed'][i+2])
                if len(cur) >= batch_size:
                    yield (self._pack_sent_tensors(prv), self._pack_sent_tensors(cur), self._pack_sent_tensors(nxt))
                    prv, cur, nxt = [], [], []

        if cur:
            yield (self._pack_sent_tensors(prv), self._pack_sent_tensors(cur), self._pack_sent_tensors(nxt))

    def get_total_samples(self):
        return len(self.data)

    def get_samples(self, batch_size, max_seq, source=None):
        steps, titles = [], []

        if not source:
            source = self.data

        for s in source:
            steps.append(s['processed'][:max_seq])
            titles.append(s['title_proc'])

            if len(titles) == batch_size:
                yield (self._pack_sample_tensors(steps, max_seq), self._pack_sent_tensors(titles))
                steps, titles = [], []

        if titles:
            yield (self._pack_sample_tensors(steps, max_seq), self._pack_sent_tensors(titles))
