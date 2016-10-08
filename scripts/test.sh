#!/bin/bash
MAIN=src/parser.py
OUTDIR=$1
MODELNUM=$2
TEST=data/ko-universal-test.conll

python $MAIN \
    --predict \
    --outdir $OUTDIR \
    --model $OUTDIR/model$MODELNUM \
    --test $TEST \
    --params $OUTDIR/params.pickle
