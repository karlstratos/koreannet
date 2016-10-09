#!/bin/bash
MAIN=src/parser.py
SEED=123456789
MEM=6000
OUTDIR=tmp/parser-word-char-jamo
TRAIN=data/ko-universal-train.conll.shuffled
DEV=data/ko-universal-dev.conll
EPOCHS=30
LSTMDIM=50
LSTMLAYERS=1

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
    --userl \
    --pembedding 0 \
    --usechar \
    --usejamo
