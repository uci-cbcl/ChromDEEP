#!/usr/bin/env python

import sys
import time

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
# from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

NB_FILTER = 300
NB_HIDDEN = 300
FILTER_LEN = 20
POOL_FACTOR = 5
DROP_OUT_CNN = 0.5
DROP_OUT_MLP = 0.5
ACTIVATION = 'relu'
# LR = 0.01
# DECAY = 1e-6
# MOMENTUM = 0.9
BATCH_SIZE = 512
NB_EPOCH = 30


def main():
    base_name = sys.argv[1]
    save_name = sys.argv[2]
    
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
    pool_len = (seq_len-FILTER_LEN+1)/POOL_FACTOR
    
    model = Sequential()
    
    model.add(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER,
                        border_mode='valid',
                        filter_length=FILTER_LEN,
                        activation=ACTIVATION))
#     model.add(MaxPooling1D(pool_length=seq_len-FILTER_LEN))
    model.add(MaxPooling1D(pool_length=pool_len, stride=pool_len))
    model.add(Dropout(DROP_OUT_CNN))
    model.add(Flatten())
    
    model.add(Dense(NB_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT_MLP))
    
    model.add(Dense(input_dim=NB_HIDDEN, output_dim=class_num))
    model.add(Activation('softmax'))
     
#     sgd = SGD(lr=LR, decay=DECAY, momentum=MOMENTUM)
 
    print 'model compiling...'
    sys.stdout.flush()
     
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode='categorical')
    
    checkpointer = ModelCheckpoint(filepath=save_name+'.hdf5', verbose=1, save_best_only=True)
#    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    outmodel = open(save_name+'.json', 'w')
    outmodel.write(model.to_json())
    outmodel.close()
    
    print 'training...'
    sys.stdout.flush()
    
    time_start = time.time()
    model.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, 
              show_accuracy=True, validation_data=(X_va, Y_va),
              callbacks=[checkpointer])
    time_end = time.time()
    
    model.load_weights(save_name+'.hdf5')
    Y_va_hat = model.predict(X_va, BATCH_SIZE, verbose=1)
    Y_te_hat = model.predict(X_te, BATCH_SIZE, verbose=1)
    acc_va = np.where(Y_va.argmax(axis=1) == Y_va_hat.argmax(axis=1))[0].size*1.0/Y_va.shape[0]
    acc_te = np.where(Y_te.argmax(axis=1) == Y_te_hat.argmax(axis=1))[0].size*1.0/Y_te.shape[0]


    print '*'*100
    print '%s accuracy_va : %.4f' % (base_name, acc_va)
    print '%s accuracy_te : %.4f' % (base_name, acc_te)
    print '%s training time : %d sec' % (base_name, time_end-time_start)
    
if __name__ == '__main__':
    main()


