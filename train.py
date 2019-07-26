
# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab, load_dev_data
from modules import *
import os, codecs
from tqdm import tqdm

import numpy as np
import codecs
import nltk
class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.x_length,self.y, self.num_batch,self.source,self.target = get_batch_data() # (N, T)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None,hp.max_turn,hp.maxlen))
                self.x_length = tf.placeholder(tf.int32,shape=(None,hp.max_turn))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

            # Load vocabulary    
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()
            
            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                embeddingsize = hp.hidden_units/2
                self.enc_embed = embedding(tf.reshape(self.x,[-1,hp.maxlen]), 
                                      vocab_size=len(de2idx), 
                                      num_units=embeddingsize, 
                                      scale=True,
                                      scope="enc_embed")
                single_cell = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
                self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell]*hp.num_layers)
                print (self.enc_embed.get_shape())
                self.sequence_length=tf.reshape(self.x_length,[-1])
                print(self.sequence_length.get_shape())
                self.uttn_outputs, self.uttn_states = tf.nn.dynamic_rnn(cell=self.rnn_cell, inputs=self.enc_embed,sequence_length=self.sequence_length, dtype=tf.float32,swap_memory=True)
                self.enc = tf.reshape(self.uttn_states,[hp.batch_size,hp.max_turn,hp.hidden_units])
                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                else:
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                 
                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc,_ = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
            
            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(self.decoder_inputs, 
                                      vocab_size=len(en2idx), 
                                      num_units=hp.hidden_units,
                                      scale=True, 
                                      scope="dec_embed")
                
                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec,_ = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec,self.attn = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
                
            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(en2idx))
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
                
            if is_training:  
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

if __name__ == '__main__':                
    # Load vocabulary    
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    X,X_length,Y, Sources, Targets = load_dev_data()
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    #preEpoch= 
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with sv.managed_session(config = tfconfig) as sess:
        early_break = 0
        old_eval_loss=10000
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            loss=[]
            
            if early_break >=4:
                break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _,loss_step,attns,sources,targets = sess.run([g.train_op,g.mean_loss,g.attn,g.source,g.target])
                loss.append(loss_step)
                
                if step%2000==0:
                    gs = sess.run(g.global_step)
                    print("train loss:%.5lf\n"%(np.mean(loss)))
                    sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

                    mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]
                    fout = codecs.open( mname, "w","utf-8")
                    eval_loss=[]
                    bleu=[]
                  
                    for i in range(len(X) // hp.batch_size):
                       ### Get mini-batches
                       x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                       x_length=X_length[i*hp.batch_size: (i+1)*hp.batch_size]
                       y = Y[i*hp.batch_size: (i+1)*hp.batch_size]
                       sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                       targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                       eval_bath = sess.run(g.mean_loss, {g.x: x,g.x_length:x_length,g.y: y})
                       eval_loss.append( eval_bath)
                       
                       preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                       for j in range(hp.maxlen):
                           _preds = sess.run(g.preds, {g.x: x,g.x_length:x_length, g.y: preds})
                           preds[:, j] = _preds[:, j]

                    
                    
                       ### Write to file
                       list_of_refs, hypotheses = [], []
                       for source, target, pred in zip(sources, targets, preds): # sentence-wise
                           got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                           fout.write("- source: " + source +"\n")
                           fout.write("- expected: " + target + "\n")
                           fout.write("- got: " + got + "\n\n")
                           fout.flush()
                           # bleu score
                           ref = target.split()
                           hypothesis = got.split()
                           score = nltk.translate.bleu_score.sentence_bleu([hypothesis],ref,(0.25, 0.25, 0.25, 0.25),nltk.translate.bleu_score.SmoothingFunction().method1)
                           bleu.append(score)
                    fout.write("train loss = %.5lf\teval loss = %.5lf\tBleu Score = %.5lf\n" %(np.mean(loss),np.mean(eval_loss),100*np.mean(bleu)))
                    print("eval loss:%.5lf"%(np.mean(eval_loss)))
                    print("Bleu Score:%.5lf"%(100*np.mean(bleu)))
                    if np.mean(eval_loss) > old_eval_loss:
                        early_break +=1
                    else:
                        early_break = 0
                    old_eval_loss=np.mean(eval_loss)
                    if early_break>=4:
                        break
                    #attention analysis
                    break
    print("Done")    
    

