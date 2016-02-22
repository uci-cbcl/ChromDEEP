#!/usr/bin/env python

import sys
import time

import numpy as np
    
from keras.models import Graph 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.constraints import maxnorm, nonneg
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

np.random.seed(0)

FILTER_LEN1 = 10
FILTER_LEN2 = 20
FILTER_LEN3 = 30
NB_FILTER1 = 50
NB_FILTER2 = 200
NB_FILTER3 = 50
POOL_FACTOR = 1
NB_HIDDEN = 15
DROP_OUT_CNN = 0.2
DROP_OUT_MLP = 0.1
# ACTIVATION = 'relu'
BATCH_SIZE = 512
NB_EPOCH = 50



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
    pool_len1 = (seq_len-FILTER_LEN1+1)/POOL_FACTOR
    pool_len2 = (seq_len-FILTER_LEN2+1)/POOL_FACTOR
    pool_len3 = (seq_len-FILTER_LEN3+1)/POOL_FACTOR
    
    model = Graph()
    
    model.add_input(name='input', input_shape=(seq_len, channel_num))
    
    #convolution layer 1
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER1,
                        border_mode='valid',
                        filter_length=FILTER_LEN1,
                        activation='relu'),
                   name='conv1', input='input')
    model.add_node(MaxPooling1D(pool_length=pool_len1, stride=pool_len1), name='maxpool1', input='conv1')
    model.add_node(Dropout(DROP_OUT_CNN), name='drop_cnn1', input='maxpool1')
    model.add_node(Flatten(), name='flat1', input='drop_cnn1')
    
    #convolution layer 2
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER2,
                        border_mode='valid',
                        filter_length=FILTER_LEN2,
                        activation='relu'),
                   name='conv2', input='input')
    model.add_node(MaxPooling1D(pool_length=pool_len2, stride=pool_len2), name='maxpool2', input='conv2')
    model.add_node(Dropout(DROP_OUT_CNN), name='drop_cnn2', input='maxpool2')
    model.add_node(Flatten(), name='flat2', input='drop_cnn2')
    
    #convolution layer 3
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=NB_FILTER3,
                        border_mode='valid',
                        filter_length=FILTER_LEN3,
                        activation='relu'),
                   name='conv3', input='input')
    model.add_node(MaxPooling1D(pool_length=pool_len3, stride=pool_len3), name='maxpool3', input='conv3')
    model.add_node(Dropout(DROP_OUT_CNN), name='drop_cnn3', input='maxpool3')
    model.add_node(Flatten(), name='flat3', input='drop_cnn3')


    model.add_node(Dense(NB_HIDDEN), name='dense1', inputs=['flat1', 'flat2', 'flat3'])
    model.add_node(Activation('relu'), name='act1', input='dense1')
    model.add_node(Dropout(DROP_OUT_MLP), name='drop_mlp1', input='act1')
    
    model.add_node(Dense(input_dim=NB_HIDDEN, output_dim=class_num, b_constraint=maxnorm(0)), name='dense2', input='drop_mlp1')
    model.add_node(Activation('sigmoid'), name='act2', input='dense2')
    
    model.add_output(name='output', input='act2')
          
    print 'model compiling...'
    sys.stdout.flush()
     
    model.compile(loss={'output':'binary_crossentropy'}, optimizer='rmsprop')
    
    checkpointer = ModelCheckpoint(filepath=save_name+'.hdf5', verbose=1, save_best_only=True)

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
    n_va = Y_va.shape[0]
    n_te = Y_te.shape[0]
    Y_va_hat = np.round(model.predict({'input':X_va}, BATCH_SIZE, verbose=1)['output'])
    Y_te_hat = np.round(model.predict({'input':X_te}, BATCH_SIZE, verbose=1)['output'])
    Y_va_0_percent = 1-Y_va.sum(axis=0)/n_va
    Y_te_0_percent = 1-Y_te.sum(axis=0)/n_te
    acc_va = 1-np.abs(Y_va-Y_va_hat).sum(axis=0)/n_va
    acc_te = 1-np.abs(Y_te-Y_te_hat).sum(axis=0)/n_te

    
    print '*'*120
    print '%s col_name :\t%s' % (base_name, '\t'.join(['CTCF', 'POL2', 'DUKE', 'FAIRE', 'UW', '27me3', 
                                                       '36me3', '4me3', '27ac', '4me1', '4me2', '9ac', '20me1']))
    print '%s Y_0%%_va :\t%s' % (base_name, '\t'.join((map(lambda x:'{0:.4f}'.format(x), Y_va_0_percent))))
    print '%s acc_va :\t%s' % (base_name, '\t'.join((map(lambda x:'{0:.4f}'.format(x), acc_va))))
    print '%s Y_0%%_te :\t%s' % (base_name, '\t'.join((map(lambda x:'{0:.4f}'.format(x), Y_te_0_percent))))
    print '%s acc_te :\t%s' % (base_name, '\t'.join((map(lambda x:'{0:.4f}'.format(x), acc_te))))
    print '%s acc_va_mean :\t%.4f\tY_0%%_va_mean :\t%.4f' % (base_name, acc_va.mean(), Y_va_0_percent.mean())
    print '%s acc_te_mean :\t%.4f\tY_0%%_te_mean :\t%.4f' % (base_name, acc_te.mean(), Y_te_0_percent.mean())
    print '%s training time : %d sec' % (base_name, time_end-time_start)
    
if __name__ == '__main__':
    main()


