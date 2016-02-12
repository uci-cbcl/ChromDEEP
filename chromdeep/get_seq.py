#!/usr/bin/env python

import sys

import pysam

BIN = 200

def main():
    infasta = pysam.Fastafile(sys.argv[1])
    infile = open(sys.argv[2])
    flank = int(sys.argv[3])
    
    
    for line in infile:
        chr, start, end, state, color = line.strip('\n').split('\t')[0:5]
        start = int(start)
        end = int(end)
        seq = infasta.fetch(chr, max(0, start-flank), end+flank).upper()
        
        if len(seq) < BIN+flank*2:
            seq += 'N'*(BIN+flank*2-len(seq))
        
        print line.strip('\n')+'\t'+seq
        #print '\t'.join([chr, str(start), str(end), state, color, seq])
    
    
    
    infile.close()
    infasta.close()
    
    
if __name__ == '__main__':
    main()

