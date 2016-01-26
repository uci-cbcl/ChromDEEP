#!/usr/bin/env python

import sys

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD

NB_FILTER = 100
NB_HIDDEN = 500
FILTER_LEN = 20
DROP_OUT_CNN = 0.25
DROP_OUT_MLP = 0.5
ACTIVATION = 'relu'
LR = 0.01
DECAY = 1e-6
MOMENTUM = 0.9
BATCH_SIZE = 100
NB_EPOCH = 50


def main():
    base_name = sys.argv[1]
    
    print 'loading data...'
    sys.stdout.flush()
    
    X_tr = np.load('X_'+base_name+'_tr_float32.npy')
    Y_tr = np.load('Y_'+base_name+'_tr_float32.npy')
    X_va = np.load('X_'+base_name+'_va_float32.npy')
    Y_va = np.load('Y_'+base_name+'_va_float32.npy')
    X_te = np.load('X_'+base_name+'_te_float32.npy')
    Y_te = np.load('Y_'+base_name+'_te_float32.npy')
    
    __, seq_len, channel_num = X_tr.shape
    __, class_num = Y_tr.shape
    
    model = Sequential()
    
    model.add(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER,
                        filter_length=FILTER_LEN,
                        activation=ACTIVATION))
    model.add(MaxPooling1D(pool_length=seq_len-FILTER_LEN))
    model.add(Dropout(DROP_OUT_CNN))
    model.add(Flatten())
    
    model.add(Dense(input_dim=NB_FILTER, output_dim=NB_HIDDEN))
#     model.add(Dense(NB_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT_MLP))
    
    model.add(Dense(input_dim=NB_HIDDEN, output_dim=class_num))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=LR, decay=DECAY, momentum=MOMENTUM)

    print 'model compiling...'
    sys.stdout.flush()
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    print 'training...'
    sys.stdout.flush()
    
    model.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, 
              show_accuracy=True, validation_data=(X_va, Y_va))
    
    loss_te, acc_te = model.evaluate(X_te, Y_te, show_accuracy=True)
    
    print '*'*100
    print 'accuracy_te : %s' % (acc_te)
    
if __name__ == '__main__':
    main()


