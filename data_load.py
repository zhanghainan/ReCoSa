# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        source_sent_split = source_sent.split(u"</d>")
        x=[]
        for sss in source_sent_split:
            if len(sss.split())==0:
                continue
            x.append( [de2idx.get(word, 1) for word in (sss + u" </S>").split()]) # 1: OOV, </S>: End of Text
        target_sent_split = target_sent.split(u"</d>")
        y = [en2idx.get(word, 1) for word in (target_sent_split[0] + u" </S>").split()] 
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list),hp.max_turn, hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    X_length=np.zeros([len(x_list),hp.max_turn],np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        for j in range(len(x)):
            if j >= hp.max_turn :
                break
            if len(x[j])<hp.maxlen:
                X[i][j] = np.lib.pad(x[j], [0, hp.maxlen-len(x[j])], 'constant', constant_values=(0, 0))
            else:
                X[i][j]=x[j][:hp.maxlen]
            X_length[i][j] = len(x[j])
        #X[i] = X[i][:len(x)]
        #X_length[i] = X_length[i][:len(x)]
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X,X_length, Y, Sources, Targets

def load_train_data():
    de_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line]
    
    X,X_length, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, X_length,Y,Sources,Targets
    
def load_test_data():
    def _refine(line):
        #line = regex.sub("<[^>]+>", "", line)
        #line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]
        
    X, X_length, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X,X_length, Sources, Targets # (1064, 150)

def load_dev_data():
    def _refine(line):
        #line = regex.sub("<[^>]+>", "", line)
        #line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_dev, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [_refine(line) for line in codecs.open(hp.target_dev, 'r', 'utf-8').read().split("\n") if line]
        
    X, X_length, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X,X_length,Y, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X,X_length, Y, sources,targets = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    X_length = tf.convert_to_tensor(X_length,tf.int32)
    # Create Queues
    input_queues = tf.train.slice_input_producer([X,X_length, Y,sources,targets])
            
    # create batch queues
    x,x_length, y,sources,targets = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x,x_length, y, num_batch ,sources,targets# (N, T), (N, T), ()

