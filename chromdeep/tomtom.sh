#!/bin/bash

~/Programs/MEME/meme_4.11.1/bin/tomtom -no-ssc -oc ./$2 -verbosity 2 -min-overlap 5 -dist pearson -evalue -thresh 0.01 $1 ~/Programs/MEME/meme_4.11.1/db/jolma2013.meme ~/Programs/MEME/meme_4.11.1/db/JASPAR_CORE_2016_vertebrates.meme ~/Programs/MEME/meme_4.11.1/db/uniprobe_mouse.meme

