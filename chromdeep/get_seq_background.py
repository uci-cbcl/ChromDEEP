#!/usr/bin/env python

import sys
import numpy as np


def main():
    inbed = open(sys.argv[1])
    
    for line in inbed:
        fields = line.strip('\n').split('\t')
        marks = fields[5:19]
        seq = fields[-1]
        
        if '1' in marks or 'N' in seq:
            continue
        
        print line.strip('\n')

    
    
    inbed.close()
    
    
    
if __name__ == '__main__':
    main()

