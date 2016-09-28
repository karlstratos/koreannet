python src/parser.py --cnn-seed 123456789 --outdir /tmp/ --model /tmp/small_orig --train data/ko-universal-train.conll.shuffled.small --dev data/ko-universal-train.conll.shuffled.small --epochs 10 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl

python src/parser.py --cnn-seed 123456789 --outdir /tmp/ --model /tmp/small_char --train data/ko-universal-train.conll.shuffled.small --dev data/ko-universal-train.conll.shuffled.small --epochs 10 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --usechar

python3 get_jamo.py vocab.txt jamos.txt

# Original performance (with POS embeddings)
python src/parser.py --cnn-seed 123456789 --outdir /tmp/orig --model /tmp/orig --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 25

# Original performance (without POS embeddings)
python src/parser.py --cnn-seed 123456789 --outdir /tmp/orig_nopos --model /tmp/orig_nopos --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0

# Character-based performance (with POS embeddings)
python src/parser.py --cnn-seed 123456789 --outdir /tmp/char --model /tmp/char --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 25 --usechar --cembedding 25

# Character-based performance (without POS embeddings)
python src/parser.py --cnn-seed 123456789 --outdir /tmp/char_nopos --model /tmp/char_nopos --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0 --usechar --cembedding 25

# Character CONCAT jamos ====> char-LSTMs performance (with POS embeddings)
python src/parser.py --cnn-seed 123456789 --outdir /tmp/charCONCATjamo --model /tmp/charCONCATjamo --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 25 --usechar --cembedding 25 --jamos jamos.txt

# Character CONCAT jamos ====> char-LSTMs performance (without POS embeddings)
python src/parser.py --cnn-seed 123456789 --outdir /tmp/charCONCATjamo_nopos --model /tmp/charCONCATjamo_nopos --train data/ko-universal-train.conll.shuffled --dev data/ko-universal-dev.conll --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm --k 3 --usehead --userl --pembedding 0 --usechar --cembedding 25 --jamos jamos.txt
