#!/bin/bash
MAIN=src/parser.py
OUTDIR=$1
MODELNUM=$2
TEST=$3

python $MAIN \
    --predict \
    --outdir $OUTDIR \
    --model $OUTDIR/model$MODELNUM \
    --test $TEST
