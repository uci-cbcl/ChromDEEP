#!/usr/bin/env python

import sys
from random import random

P_TR = 0.70
P_VA = 0.15
P_TE = 0.15
CHUNK_SIZE = 10000
BIN_SIZE = 200
FLANK_SIZE = 400
BUCKET_SIZE = CHUNK_SIZE/BIN_SIZE
GAP_SIZE = FLANK_SIZE/BIN_SIZE

def main():
    inbed = open(sys.argv[1])
    outbase = sys.argv[2]
    outtr = open(outbase+'_tr.bed', 'w')
    outva = open(outbase+'_va.bed', 'w')
    outte = open(outbase+'_te.bed', 'w')
    
    lines = inbed.readlines()
    line_num = len(lines)
    
    i = 0
    while i+BUCKET_SIZE < line_num:
        p = random()
        
        if p < P_TR:
            outtr.write(''.join(lines[i+GAP_SIZE:i+BUCKET_SIZE-GAP_SIZE]))
        elif p < P_TR+P_VA:
            outva.write(''.join(lines[i+GAP_SIZE:i+BUCKET_SIZE-GAP_SIZE]))
        else:
            outte.write(''.join(lines[i+GAP_SIZE:i+BUCKET_SIZE-GAP_SIZE]))
    
        i += BUCKET_SIZE

    
    
    inbed.close()
    outtr.close()
    outva.close()
    outte.close()
    
    
    
if __name__ == '__main__':
    main()

