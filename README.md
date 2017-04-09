
# Testing
python src/parser.py --predict --outdir /tmp/orig_nopos --model /tmp/orig_nopos/themodel25 --test data/ko-universal-test.conll --params /tmp/orig_nopos/params.pickle




# Word (without POS embeddings)
python src/parser.py --dynet-mem 2000 --dynet-seed 123456789 --outdir /tmp/orig_nopos --model /tmp/orig_nopos/themodel --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0

# Word + Char (without POS embeddings)
python src/parser.py --dynet-mem 2000 --dynet-seed 123456789 --outdir /tmp/origchar_nopos --model /tmp/origchar_nopos/themodel --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0 --usechar --cembedding 25

# Word + Char + Jamo (without POS embeddings)
python src/parser.py --dynet-mem 2000 --dynet-seed 123456789 --outdir /tmp/origcharjamo_nopos --model /tmp/origcharjamo_nopos/themodel --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0 --usechar --cembedding 25 --usejamo

# Char (without POS embeddings)
python src/parser.py --dynet-mem 4000 --dynet-seed 123456789 --outdir /tmp/char_nopos --model /tmp/char_nopos/themodel --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0 --noword --cembedding 50 --usechar

# Jamo (without POS embeddings)
python src/parser.py --dynet-mem 4000 --dynet-seed 123456789 --outdir /tmp/jamo_nopos --model /tmp/jamo_nopos/themodel --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0 --noword --cembedding 50 --usejamo
