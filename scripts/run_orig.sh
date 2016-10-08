#!/bin/bash
MAIN=src/parser.py
SEED=123456789
MEM=512
OUTDIR=tmp/parser-orig
TRAIN=data/ko-universal-train.conll.shuffled
DEV=data/ko-universal-dev.conll
EPOCHS=30
LSTMDIM=125
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
