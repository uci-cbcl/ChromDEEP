#!/usr/bin/env python

import sys

import numpy as np

from chromdeep.utils import *

SEQ_LEN = 600
CLASS_NUM = 13

def main():
    infile = open(sys.argv[1])
    outbase = sys.argv[2]

    lines = infile.readlines()
    line_num = len(lines)
    
    X = np.zeros((line_num, SEQ_LEN, 4), dtype='float32')
    Y = np.zeros((line_num, CLASS_NUM), dtype='float32')
    
    for i in range(0, line_num):
        line = lines[i]
        fields = line.strip('\n').split('\t')
        marks = fields[5:19]
        seq = fields[-1]
        
        x = onehot(seq)
        y = np.array(marks[0:12]+marks[13:14]) #no Control mark
        
        X[i, :, :] = x
        Y[i, :] = y
    
        i += 1
        
        if i%10000 == 0:
            print '%s/%s lines processed...' % (i, line_num)
    
    
    np.save('X_'+outbase+'_float32.npy', X)
    np.save('Y_'+outbase+'_float32.npy', Y)
    infile.close()


if __name__ == '__main__':
    main()



