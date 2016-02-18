#!/usr/bin/env python

import sys
import os
import shutil

COL_NAME = ['CTCF', 'POL2', 'DUKE_DNASE', 'FAIRE', 'UW_DNASE', 'H3K27me3', 'H3K36me3', 
            'H3K4me3', 'H3K27ac', 'H3K4me1', 'H3K4me2', 'H3K9ac', 'Control', 'H4K20me1']



def main():
    infile = open(sys.argv[1])
    outdir = sys.argv[2]
    cell_type = sys.argv[3]
    
    if os.path.exists('./'+outdir):
        shutil.rmtree('./'+outdir)
    
    os.mkdir('./'+outdir)
    
    chrom_prev = 'chr1'
    outfile = open('./'+outdir+'/'+chrom_prev+'_binary.txt', 'w')
    outfile.write(cell_type+'\t'+chrom_prev+'\n')
    outfile.write('\t'.join(COL_NAME[0:12]+COL_NAME[13:14])+'\n')
    
    for line in infile:
        fields = line.strip('\n').split('\t')
        chrom = fields[0]
        marks = fields[5:19]
        
        if chrom != chrom_prev:
            chrom_prev = chrom
            outfile.close()
            outfile = open('./'+outdir+'/'+chrom_prev+'_binary.txt', 'w')
            outfile.write(cell_type+'\t'+chrom_prev+'\n')
            outfile.write('\t'.join(COL_NAME[0:12]+COL_NAME[13:14])+'\n')
        
        outfile.write('\t'.join(marks[0:12]+marks[13:14])+'\n')
    
    outfile.close()
    
    
if __name__ == '__main__':
    main()
