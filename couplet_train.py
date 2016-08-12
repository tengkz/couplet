# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:41:41 2016

@author: hztengkezhen
"""

import seq2seq
from seq2seq.models import Seq2seq
import numpy as np
import pickle


        
print "read pkl files ..."
with open('word5_dim8_dict.pkl','rb') as f:
    token2vec = pickle.load(f)
with open('train.pkl','rb') as f:
    [X,Y] = pickle.load(f)
    X = X[0:141664,:,:]
    Y = Y[0:141664,:,:]
print "read pkl files done!"

# output_dim = 8
# hidden_dim = 16
# output_length = 5

print "build model ..."
model = Seq2seq(8,16,5,depth=2,batch_input_shape=(32,5,8))
model.compile(loss='mse',optimizer='rmsprop')
print "build model done!"

print "model fit begin ..."
model.fit(X,Y,nb_epoch=1,batch_size=32)
print "model fit done!"

model.save('model1.h5')

