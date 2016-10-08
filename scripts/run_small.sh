#!/bin/bash
MAIN=src/parser.py
SEED=123456789
MEM=512
OUTDIR=tmp/parser-small
TRAIN=data/ko-universal-train.conll.shuffled.small
DEV=data/ko-universal-train.conll.shuffled.small
EPOCHS=1
LSTMDIM=50
LSTMLAYERS=2

mkdir -p $OUTDIR

python $MAIN \
    --cnn-mem $MEM \
    --cnn-seed $SEED \
    --outdir $OUTDIR \
    --model model \
    --train $TRAIN \
    --dev $DEV \
    --epochs $EPOCHS \
    --lstmdims $LSTMDIM \
    --lstmlayers $LSTMLAYERS \
    --bibi-lstm \
    --k 3 \
    --usehead \
    --userl

python $MAIN \
    --cnn-mem $MEM \
    --cnn-seed $SEED \
    --predict \
    --outdir $OUTDIR \
    --model $OUTDIR/model$EPOCHS \
    --params $OUTDIR/params.pickle \
    --test $DEV

head -3 $OUTDIR/test_pred.conll.txt
