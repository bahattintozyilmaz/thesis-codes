import math

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Sequential
from keras.layers import GRU, Dense, Activation, Dropout, LSTM, Bidirectional, Embedding, TimeDistributed, RepeatVector

from utils import embed_encode, embed_generator, load_data, top_k_accuracy_gen, test_sentence
from keras_contrib import mLSTM

data_path = '/Users/baha/Personal/thesis/wikihowdumpall.clean.json'
out_path = '/Users/baha/Personal/thesis/nn-models/pred-char-20-epoch'

embedding_size = 50
lookback = 100
batch_size = 128

try:
	from overrides import*
except ImportError:
	pass

print('loading data')
x_train, x_test = load_data(data_path, filter_cats=['Home Improvements and Repairs'])

all_chars = set([c for c in x_test+x_train])
char_dict = {c:i for i,c in enumerate(sorted(list(all_chars)))}

print('data loaded. test {}, train {}'.format(len(x_test), len(x_train)))

with open(out_path + '.charmap', 'w') as f:
	import json
	json.dump(char_dict, f)

sent_autoencoder = Sequential()
sent_autoencoder.add(Embedding(len(all_chars), embedding_size))
sent_autoencoder.add(mLSTM(300, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, implementation=2,
						   activity_regularizer='l2', activation='tanh'))
sent_autoencoder.add(Dense(len(all_chars), activation='softmax', activity_regularizer='l1_l2'))

sent_autoencoder.compile(loss='categorical_crossentropy', optimizer='adam', 
						 metrics=['categorical_crossentropy',
								  'mean_squared_error', top_k_accuracy_gen(3), 'accuracy'])

epoch_steps = math.ceil((len(x_train)-lookback-1)/float(batch_size))
validation_steps = math.ceil((len(x_test)-lookback-1)/float(batch_size))

sent_autoencoder.summary()

hist = sent_autoencoder.fit_generator(
	embed_generator(x_train, batch_size, lookback, char_dict), epoch_steps, epochs=20, 
	callbacks=[ModelCheckpoint(out_path+'.{epoch:02d}-{val_loss:.4f}.hdf5'),
			   CSVLogger(out_path+'.loss.log')],
	validation_data=embed_generator(x_test, batch_size, lookback, char_dict), 
	validation_steps=validation_steps)

print(hist.__dict__)

sent_autoencoder.save(out_path + '.h5')