#!/usr/bin/env python

import sys

import numpy as np

from chromdeep.utils import *

# TARGET = {'BR':0, 'O':1, 'B':2, 'G':3}
SEQ_LEN = 1000

def main():
    infile = open(sys.argv[1])
    outbase = sys.argv[2]
    color_str = sys.argv[3] #'BR,B,O,G'

    lines = infile.readlines()
    line_num = len(lines)
    color_lst = color_str.split(',')
    color_dict = {}
    class_num = len(color_lst)
    
    for k in range(0, class_num):
        color_dict[color_lst[k]] = k
    
    X = np.zeros((line_num, SEQ_LEN, 4), dtype='float32')
    Y = np.zeros((line_num, class_num), dtype='float32')
    
    for i in range(0, line_num):
        line = lines[i]
        chrom, start, end, label, color, seq = line.strip('\n').split('\t')
        x = onehot(seq)
        y_idx = color_dict[color]
        
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



