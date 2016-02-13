#!/usr/bin/env python

import sys
import numpy as np


def main():
    inbed = open(sys.argv[1])
    
    for line in inbed:
        fields = line.strip('\n').split('\t')
        marks = fields[5:19]
                
        #Control mark == 1
        if marks[12] == '1':
            continue
        
        if '1' not in marks:
            continue
        
        print line.strip('\n')

    
    
    inbed.close()
    
    
    
if __name__ == '__main__':
    main()

