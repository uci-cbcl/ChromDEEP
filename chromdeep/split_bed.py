#!/usr/bin/env python

import sys
import numpy as np

np.random.seed(0)

P_TR = 0.70
P_VA = 0.15
P_TE = 0.15
CHUNK_SIZE = 10000
BIN_SIZE = 200
#FLANK_SIZE = 200
BUCKET_SIZE = CHUNK_SIZE/BIN_SIZE
#GAP_SIZE = FLANK_SIZE/BIN_SIZE

def main():
    inbed = open(sys.argv[1])
    outbase = sys.argv[2]
    flank_size = int(sys.argv[3])

    gap_size = flank_size/BIN_SIZE
    outtr = open(outbase+'_tr.bed', 'w')
    outva = open(outbase+'_va.bed', 'w')
    outte = open(outbase+'_te.bed', 'w')
    
    lines = inbed.readlines()
    line_num = len(lines)
    
    i = 0
    while i+BUCKET_SIZE < line_num:
        p = np.random.rand()
        
        if p < P_TR:
            outtr.write(''.join(lines[i+gap_size:i+BUCKET_SIZE-gap_size]))
        elif p < P_TR+P_VA:
            outva.write(''.join(lines[i+gap_size:i+BUCKET_SIZE-gap_size]))
        else:
            outte.write(''.join(lines[i+gap_size:i+BUCKET_SIZE-gap_size]))
    
        i += BUCKET_SIZE

    
    
    inbed.close()
    outtr.close()
    outva.close()
    outte.close()
    
    
    
if __name__ == '__main__':
    main()

