#!/usr/bin/env python

import sys

import numpy as np
from matplotlib import pyplot as plt

NAR_name = ['H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3', 
            'H4K20me1', 'POL2', 'CTCF', 'DUKE_DNASE', 'UW_DNASE', 'FAIRE']

def main():
    infile = open(sys.argv[1])
    outfig = sys.argv[2]
    
    col_name = infile.next().strip('\n').split('\t')[1:]
    emission = []
    
    for line in infile:
        emission.append(map(float, line.strip('\n').split('\t')[1:]))
    
    emission = np.array(emission).transpose()
    col_num, state_num = emission.shape
    
    plt.Figure(figsize=(25,20))
    plt.pcolor(emission, vmin=0.0, vmax=1.0, cmap='Blues')
    plt.xlim(0, state_num)
    plt.ylim(0, col_num)
    plt.gca().set_xticks(np.arange(emission.shape[1])+0.5, minor=False)
    plt.gca().set_yticks(np.arange(emission.shape[0])+0.5, minor=False)
    plt.gca().set_xticklabels(range(1, state_num+1), minor=False)
    plt.gca().set_yticklabels(NAR_name, rotation='horizontal', minor=False)
    plt.gca().invert_yaxis()
    plt.savefig(outfig, bbox_inches='tight')
    
    infile.close()
    
    
    
    
if __name__ == '__main__':
    main()

