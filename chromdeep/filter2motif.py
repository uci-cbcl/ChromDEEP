#!/usr/bin/env python

import sys

import numpy as np
import theano
from keras.models import model_from_json

BATCH_SIZE = 512
MEME_HEADER = """MEME version 4.4

ALPHABET= ACGT

strands: + -

Background letter frequencies (from web form):
A 0.25000 C 0.25000 G 0.25000 T 0.25000 

"""



def update_counts(counts, x, a):
    a_max = a.max(axis=1)
    a_max_idx = a.argmax(axis=1)
    
    n, seq_len, channel_num = x.shape
    nb_filter, filter_len, channel_num = counts.shape
    
    for i in range(0, n):
        for j in range(0, nb_filter):
            idx = a_max_idx[i, j]
            counts[j] += a_max[i, j]*x[i, idx:idx+filter_len, :]
    
    return counts
    


def main():
    X = np.load(sys.argv[1])
    base_name = sys.argv[2]
    
    base_json = base_name+'.json'
    base_hdf5 = base_name+'.hdf5'
    base_meme = base_name+'.meme'
    
    model = model_from_json(open(base_json).read())
    model.load_weights(base_hdf5)
    
    N, seq_len, channel_num = X.shape
    _,  act_len, nb_filter = model.layers[0].output_shape
    nb_filter, channel_num, filter_len, _ = model.layers[0].get_weights()[0].shape
        
    f = theano.function([model.get_input()], model.layers[0].get_output())
    
    counts = np.zeros((nb_filter, filter_len, channel_num))+1e-5
    
    i = 0
    while i+BATCH_SIZE < N:
        x = X[i:i+BATCH_SIZE]
        a = f(x)
        counts = update_counts(counts, x, a)
        
        i += BATCH_SIZE
        
        print '%s/%s data points processed...' % (i, N)
        sys.stdout.flush()
        
    
    x = X[i:N]
    a = f(x)
    counts = update_counts(counts, x, a)
    pwm = counts/counts.sum(axis=2).reshape(nb_filter, filter_len, 1)
    
    outfile = open(base_meme, 'w')
    outfile.write(MEME_HEADER)
    
    for i in range(0, nb_filter):
        outfile.write('MOTIF FILTER_%s\n\n' % (i))
        outfile.write('letter-probability matrix: alength= 4\n')
        
        for j in range(0, filter_len):
            outfile.write('%f\t%f\t%f\t%f\n' % tuple(pwm[i, j, :].tolist()))
        
        outfile.write('\n')
        
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    


