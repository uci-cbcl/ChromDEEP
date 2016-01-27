#!/usr/bin/env python

import sys

BIN = 200

def main():
    infile = open(sys.argv[1])
    
    for line in infile:
        chr, start, end, state = line.strip('\n').split('\t')[0:4]
        start = int(start)
        end = int(end)
        
        while start+BIN <= end:
            print '\t'.join([chr, str(start), str(start+BIN), state])
            start += BIN
    
    
    
    infile.close()
    
    
if __name__ == '__main__':
    main()

