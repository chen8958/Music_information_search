#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example code for local key detection preprocessing. You may start with filling
the '?'s below. There're also some description and hint within comment. However,
please feel free to modify anything as you like!

@author: selly
"""
import numpy as np
from glob import glob
import utils # self-defined utils.py file
from librosa.feature import chroma_stft, chroma_cqt, chroma_cens
import pretty_midi
from scipy.stats import pearsonr
DB = 'BPS_piano'

chromagram, label, index = list(), list(), list()
if(DB == 'BPS_piano'):
	for f in glob(DB+'/*.wav'):
		key = utils.parse_key([line.split('\t')[1] for line in utils.read_keyfile_bps(f).split('\n')])
		label.extend(key)
		ind = int(f.split('\\')[-1].strip('.wav'))
		if ind in utils.DATA_SPLIT[DB]['train']:
			index.extend(['train']*len(key))
		elif ind in utils.DATA_SPLIT[DB]['valid']:
			index.extend(['valid']*len(key))
		elif ind in utils.DATA_SPLIT[DB]['test']:
			index.extend(['test']*len(key))

		sr, y = utils.read_wav(f)
		for i in range(int(len(y)/sr)):
			cxx = chroma_stft(y=y[i*sr:(i+1)*sr-1], sr = sr);
			chroma_vector = cxx.sum(axis = 1);
			key_co = list()
			for j in range(0,11):
				key_co.append(pearsonr(chroma_vector,utils.rotate(utils.MODE["major"],j-3))[0]);
			for j in range(12,23):
				key_co.append(pearsonr(chroma_vector,utils.rotate(utils.MODE["minor"],j-15))[0]);
			chromagram.append(utils.LABEL[key_co.index(max(key_co))])
chromagram, label, index = map(np.array, [chromagram, label, index])
valid_x = chromagram[index=='valid']
test_x  = chromagram[index=='test' ]
train_x = chromagram[index=='train']

valid_y = label[index=='valid']
test_y  = label[index=='test' ]
train_y = label[index=='train']
'''
acc_valid = utils.same(valid_x,valid_y)/len(valid_x);
acc_test = utils.same(test_x,test_y)/len(test_x);
acc_train = utils.same(train_x,train_y)/len(train_x);
'''
acc_valid = utils.wsame(valid_x,valid_y)/len(valid_x);
acc_test = utils.wsame(test_x,test_y)/len(test_x);
acc_train = utils.wsame(train_x,train_y)/len(train_x);
print("type    \taccuracy")
print("vaild\t{:8.2f}%".format(acc_valid*100))
print("test\t{:8.2f}%".format(acc_test*100))
print("train\t{:8.2f}%".format(acc_train*100))
del chromagram, label # clean to free memory

#%%
