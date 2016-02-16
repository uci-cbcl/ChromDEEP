#!/usr/bin/env python

import sys
import time

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.constraints import maxnorm, nonneg
from keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(0)

# NB_FILTER = 200
# NB_HIDDEN = 25
FILTER_LEN = 20
POOL_FACTOR = 3
# DROP_OUT_CNN = 0.5
# DROP_OUT_MLP = 0.1
# LR = 0.01
# DECAY = 1e-6
# MOMENTUM = 0.9
BATCH_SIZE = 512
NB_EPOCH = 50


def main():
    base_name = sys.argv[1]
    save_name = sys.argv[2]
    nb_filter = int(sys.argv[3])
    nb_hidden = int(sys.argv[4])
    drop_out_cnn = float(sys.argv[5])
    drop_out_mlp = float(sys.argv[6])
    
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
                        nb_filter=nb_filter,
                        border_mode='valid',
                        filter_length=FILTER_LEN,
                        activation='relu'))
#     model.add(MaxPooling1D(pool_length=seq_len-FILTER_LEN))
    model.add(MaxPooling1D(pool_length=pool_len, stride=pool_len))
    model.add(Dropout(drop_out_cnn))
    model.add(Flatten())
    
    model.add(Dense(nb_hidden))
#     model.add(Activation('relu'))
    model.add(Activation('tanh'))
    model.add(Dropout(drop_out_mlp))
    
    model.add(Dense(input_dim=nb_hidden, output_dim=class_num, W_constraint=nonneg(), b_constraint=maxnorm(0)))
#     model.add(Dense(input_dim=nb_hidden, output_dim=class_num, b_constraint=maxnorm(0)))
    model.add(Activation('sigmoid'))
 
    print 'model compiling...'
    sys.stdout.flush()
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
    
    
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
    n_va = Y_va.shape[0]
    n_te = Y_te.shape[0]
    Y_va_hat = np.round(model.predict(X_va, BATCH_SIZE, verbose=1))
    Y_te_hat = np.round(model.predict(X_te, BATCH_SIZE, verbose=1))
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


