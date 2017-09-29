import json
import random
import numpy as np

def load_data(path, test_ratio=0.01, filter_cats=None, delimiter='\000'):
	def process_sents(sample):
		return ' '.join(s.strip() for s in sample['steps'])

	with open(path, 'r') as f:
		tokenized_data = json.load(f)

	if filter_cats:
		tokenized_data = [data for data in tokenized_data
						  if any((t in data.get('cat', [])) for t in filter_cats)]

	test_count = int(len(tokenized_data)*test_ratio)
	random.shuffle(tokenized_data)

	train_set, test_set = tokenized_data[:-test_count], tokenized_data[-test_count:]

	test_sents = delimiter + delimiter.join(process_sents(s) for s in test_set)
	train_sents = delimiter + delimiter.join(process_sents(s) for s in train_set)

	return train_sents, test_sents

def one_hot_encode(sent, j, batch_size, lookback, char_dict):
	allx = np.zeros((batch_size, lookback, len(char_dict)))
	ally = np.zeros((batch_size, len(char_dict)))
	for i in range(min(batch_size, len(sent)-j-lookback)):
		inner_sent = sent[(i+j):(i+j+lookback)]
		for ind,c in enumerate(inner_sent):
			allx[i,ind,char_dict[c]] = 1
		ally[i, char_dict[sent[i+j+lookback]]] = 1
		
	return allx.astype(np.float32), ally.astype(np.float32)

def one_hot_generator(sent, batch_size, lookback, char_dict):
	while 1:
		for j in range(0,len(sent)-lookback-1,batch_size):
			yield one_hot_encode(sent, j, batch_size, lookback, char_dict)

def embed_encode(sent, j, batch_size, lookback, char_dict):
	allx = np.zeros((batch_size, lookback))
	ally = np.zeros((batch_size, len(char_dict)))
	for i in range(min(batch_size, len(sent)-j-lookback)):
		inner_sent = sent[(i+j):(i+j+lookback)]
		for ind,c in enumerate(inner_sent):
			allx[i,ind] = char_dict[c]
		ally[i, char_dict[sent[i+j+lookback]]] = 1
		
	return allx.astype(np.float32), ally.astype(np.float32)

def embed_generator(sent, batch_size, lookback, char_dict):
	while 1:
		for j in range(0,len(sent)-lookback-1,batch_size):
			yield embed_encode(sent, j, batch_size, lookback, char_dict)

def top_k_accuracy_gen(k=3):
	from keras import backend as K

	def top_k_accuracy(y_true, y_pred):
		return K.mean(K.in_top_k(K.cast(y_pred, 'float32'), K.argmax(y_true, axis=-1), k), axis=-1)

	return top_k_accuracy

def sample_distribution(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

def run_lstm(model, seed, length, encoding_type, encoding_dict, temperature=1.0):
	model_inp = model.get_input_at(0).shape[1]._value or max_sentences
	seed = ((' '*model_inp)+seed)[-model_inp:].lower()
	ci = {v:k for k,v in ic.items()}
	ic = {v:k for k,v in ci.items()}
	chars = ""
	i_len = len(str(length))
	
	def _prepare_one_hot(seq):
		inp = np.zeros((100, model_inp, len(encoding_dict)))
		for i,c in enumerate(seq):
			inp[0,i,ci[c]] = 1
		return inp
	
	def _prepare_embedding(seq):
		inp = np.zeros((100, model_inp))
		for i,c in enumerate(seq):
			inp[0,i] = ci[c]
		return inp
			
	def _run_lstm(seq, prepare):
		inp = prepare(seq)
		res = model.predict(inp,batch_size=100,verbose=0)[0]
		maxes = [sample(res, temperature)]
		
		for i in range(4):
			res[maxes[-1]] = 0
			maxes.append(np.argmax(res))
		
		return maxes
		
	prepare = _prepare_one_hot if encoding_type == 'one-hot' else _prepare_embedding
	
	for i in range(length):
		seq = (seed+chars)[-model_inp:]
		c = _run_lstm(seq, prepare)
		c_all = [ic[cc] for cc in c]
		print(((' '*i_len)+str(i))[-i_len:], seq, c_all[0], c_all[1:])
		chars += c_all[0]
		
	return chars

def test_sentence(model, sent, char_dict, n=None, t=0.3):
	sent = '\000' + sent
	if not n:
		n = len(sent)

	res = run_lstm(model, sent, n, "embed", char_dict, temperature=t)
	print(sent+'|'+res)
