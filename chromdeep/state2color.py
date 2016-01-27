#!/usr/bin/env python

import sys

state2color = {'Tss':'BR', 'TssF':'BR', 'PromF':'LR', 'PromP':'P', 'Enh':'O', 'EnhF':'O',
               'EnhWF':'Y', 'EnhW':'Y', 'DnaseU':'Y', 'DnaseD':'Y', 'FaireW':'Y', 
               'CtcfO':'B', 'Ctcf':'B', 'Gen5':'DG', 'Elon':'DG', 'ElonW':'DG', 
               'Gen3':'DG', 'Pol2':'DG', 'H4K20':'DG', 'Low':'LG', 'ReprD':'G', 
               'Repr':'G', 'ReprW':'G', 'Quies':'W', 'Art':'W'}


def main():
    infile = open(sys.argv[1])
    
    
    for line in infile:
        chr, start, end, state = line.strip('\n').split('\t')[0:4]
        
        if "'" in state:
            state = state.strip("'")
        
        color = state2color[state]
        
        print '\t'.join([chr, start, end, state, color])
    
    
    
    infile.close()
    
    
if __name__ == '__main__':
    main()

