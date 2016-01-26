#!/usr/bin/env python

import sys

import numpy as np

ENCODE = {'A':0, 'C':1, 'G':2, 'T':3}

def onehot(seq):
    n = len(seq)
    
    code = np.zeros((n, 4), dtype='float32')
    
    for i in range(0, n):
        if seq[i] not in 'ACGTN':
            print 'ERROR: %s contains non-ACGTN chars' % (seq)
            sys.exit(-1)
        
        if seq[i] == 'N':
            continue
        
        code[i, ENCODE[seq[i]]] = 1.0
    
    return code
            

