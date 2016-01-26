#!/usr/bin/env python

import sys

import numpy as np

from chromdeep.utils import *

TARGET = {'BR':0, 'O':1, 'B':2, 'DG':3}
SEQ_LEN = 300

def main():
    infile = open(sys.argv[1])
    outbase = sys.argv[2]
    line_num = int(sys.argv[3])
    
    X = np.zeros((line_num, SEQ_LEN, 4), dtype='float32')
    Y = np.zeros((line_num, len(TARGET.keys())), dtype='float32')
    
    i = 0
    for line in infile:
        chrom, start, end, label, color, seq = line.strip('\n').split('\t')
        x = onehot(seq)
        y_idx = TARGET[color]
        
        X[i, :, :] = x
        Y[i, y_idx] = 1.0
    
        i += 1
        
        if i%10000 == 0:
            print '%s/%s lines processed...' % (i, line_num)
    
    
    np.save('X_'+outbase+'_float32.npy', X)
    np.save('Y_'+outbase+'_float32.npy', Y)
    infile.close()


if __name__ == '__main__':
    main()



