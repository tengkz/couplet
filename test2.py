# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:41:41 2016

@author: hztengkezhen
"""

import seq2seq
from seq2seq.models import Seq2seq
import numpy as np
import pickle

def distance(x,y):
    return sum((x-y)**2)

def nearest_token(v,token2vec):
    min_token = None
    min_distance = 999
    for key,value in token2vec.iteritems():
        dis = distance(v,value)
        if dis < min_distance:
            min_token = key
            min_distance = dis
    return min_token

def generate_input(line,token2vec):
    line = line.decode('utf-8')
    ret = np.zeros((1,5,8),dtype=np.float)
    for index,x in enumerate(line):
        try:
            v = token2vec[x]
        except KeyError:
            print "the word " + v.encode('utf-8') + " is not in our dictionary! Please select another!"
            return None
        for j in range(8):
            ret[0][index][j] = v[j]
    return ret

def generate_output(v,token2vec):
    output = []
    for i in range(5):
        t = nearest_token(v[0,i,:],token2vec)
        output.append(t)
    return "".join(output).encode('utf-8')
        
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

test_input = "举头望明月"
real_input = generate_input(test_input,token2vec)
#real_output = generate_output(real_input,token2vec)