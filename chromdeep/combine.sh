cp $1_BIN200_FLANK200_tr_foreground.seq $1_BIN200_FLANK200_tr_combined.seq
shuf $1_BIN200_FLANK200_tr_background.seq -o temp
head -n 150000 temp >> $1_BIN200_FLANK200_tr_combined.seq
shuf $1_BIN200_FLANK200_tr_combined.seq -o temp
mv temp $1_BIN200_FLANK200_tr_combined.seq

cp $1_BIN200_FLANK200_va_foreground.seq $1_BIN200_FLANK200_va_combined.seq
shuf $1_BIN200_FLANK200_va_background.seq -o temp
head -n 30000 temp >> $1_BIN200_FLANK200_va_combined.seq
shuf $1_BIN200_FLANK200_va_combined.seq -o temp
mv temp $1_BIN200_FLANK200_va_combined.seq

cp $1_BIN200_FLANK200_te_foreground.seq $1_BIN200_FLANK200_te_combined.seq
shuf $1_BIN200_FLANK200_te_background.seq -o temp
head -n 30000 temp >> $1_BIN200_FLANK200_te_combined.seq
shuf $1_BIN200_FLANK200_te_combined.seq -o temp
mv temp $1_BIN200_FLANK200_te_combined.seq

