#!/usr/bin/env python

import sys
import time

import numpy as np
    
from keras.models import Graph 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
    
FILTER_LEN1 = 10
FILTER_LEN2 = 20
FILTER_LEN3 = 30
NB_FILTER1 = 50
NB_FILTER2 = 150
NB_FILTER3 = 150
POOL_LEN = 200
NB_HIDDEN = 500
DROP_OUT_CNN = 0.5
DROP_OUT_MLP = 0.5
ACTIVATION = 'relu'
BATCH_SIZE = 500
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
    
    model = Graph()
    
    model.add_input(name='input', input_shape=(seq_len, channel_num))
    
    #convolution layer 1
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER1,
                        border_mode='same',
                        filter_length=FILTER_LEN1,
                        activation=ACTIVATION),
                   name='conv1', input='input')
    model.add_node(MaxPooling1D(pool_length=POOL_LEN, stride=POOL_LEN), name='maxpool1', input='conv1')
    model.add_node(Dropout(DROP_OUT_CNN), name='drop_cnn1', input='maxpool1')
    model.add_node(Flatten(), name='flat1', input='drop_cnn1')
    
    #convolution layer 2
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER2,
                        border_mode='same',
                        filter_length=FILTER_LEN2,
                        activation=ACTIVATION),
                   name='conv2', input='input')
    model.add_node(MaxPooling1D(pool_length=POOL_LEN, stride=POOL_LEN), name='maxpool2', input='conv2')
    model.add_node(Dropout(DROP_OUT_CNN), name='drop_cnn2', input='maxpool2')
    model.add_node(Flatten(), name='flat2', input='drop_cnn2')
    
    #convolution layer 3
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER3,
                        border_mode='same',
                        filter_length=FILTER_LEN3,
                        activation=ACTIVATION),
                   name='conv3', input='input')
    model.add_node(MaxPooling1D(pool_length=POOL_LEN, stride=POOL_LEN), name='maxpool3', input='conv3')
    model.add_node(Dropout(DROP_OUT_CNN), name='drop_cnn3', input='maxpool3')
    model.add_node(Flatten(), name='flat3', input='drop_cnn3')


    model.add_node(Dense(NB_HIDDEN), name='dense1', inputs=['flat1', 'flat2', 'flat3'])
    model.add_node(Activation('relu'), name='act1', input='dense1')
    model.add_node(Dropout(DROP_OUT_MLP), name='drop_mlp1', input='act1')
    
    model.add_node(Dense(input_dim=NB_HIDDEN, output_dim=class_num), name='dense2', input='drop_mlp1')
    model.add_node(Activation('softmax'), name='act2', input='dense2')
    
    model.add_output(name='output', input='act2')
          
    print 'model compiling...'
    sys.stdout.flush()
     
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer='rmsprop')
    
    checkpointer = ModelCheckpoint(filepath=save_name+'.hdf5', verbose=1, save_best_only=True)
#    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    outmodel = open(save_name+'.json', 'w')
    outmodel.write(model.to_json())
    outmodel.close()
    
    print 'training...'
    sys.stdout.flush()
    
    time_start = time.time()
    model.fit({'input':X_tr, 'output':Y_tr}, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, 
              verbose=1, validation_data={'input':X_va, 'output':Y_va},
              callbacks=[checkpointer])
    time_end = time.time()

    model.load_weights(save_name+'.hdf5')
    Y_va_hat = model.predict({'input':X_va}, BATCH_SIZE, verbose=1)['output']
    Y_te_hat = model.predict({'input':X_te}, BATCH_SIZE, verbose=1)['output']
    acc_va = np.where(Y_va.argmax(axis=1) == Y_va_hat.argmax(axis=1))[0].size*1.0/Y_va.shape[0]
    acc_te = np.where(Y_te.argmax(axis=1) == Y_te_hat.argmax(axis=1))[0].size*1.0/Y_te.shape[0]

    
    print '*'*100
    print '%s accuracy_va : %.4f' % (base_name, acc_va)
    print '%s accuracy_te : %.4f' % (base_name, acc_te)
    print '%s training time : %d sec' % (base_name, time_end-time_start)
    
if __name__ == '__main__':
    main()


