# KoreanNet

KoreanNet is a neural architecture for the unique compositional orthography of the Korean language. It decomposes each character to extract underlying __jamo__ letters (basic phonetic units) and induce character represenations from the letters. For example, the embedding for the character 갔 is a function of ㄱ, ㅏ, and ㅆ. The decomposition can be performed deterministically on the fly by Unicode manipulation.

This package implements KoreanNet for dependency parsing by extending the [BiLSTM parser of Kiperwasser and Goldberg (2016)](https://github.com/elikip/bist-parser) using the Korean Unicode manipulation tools implemented [here](https://github.com/JDongian/python-jamo).

### Prerequisites
- [Dynet for Python](https://github.com/clab/dynet)
- [Korean treebank in the universal treebank v2](https://github.com/ryanmcd/uni-dep-tb)

### Training Commands
See below for examples on how to train a parser. The code has disabled shuffling for reproducibility in experiments. To enable shuffling, uncomment `random.shuffle(shuffledData)` in arc_hybrid.py.

1. Word only (100 dimensional with pre-trained embeddings)
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/word --train ko-universal-train.conll.shuffled --dev ko-universal-dev.conll --extrn emb100.ko --wembedding 100
```
2. Character only (100 dimensional)
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/char --train ../data/ko-universal-train.conll.shuffled --dev ../data/ko-universal-dev.conll  --cembedding 100  --usechar --noword
```
3. Jamo only (100 dimensional)
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/jamo --train ../data/ko-universal-train.conll.shuffled --dev ../data/ko-universal-dev.conll  --cembedding 100  --usejamo --noword
```
4. Word (100 dimensional with pre-trained embeddings), character and jamo (100 dimensional)
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/word-char-jamo --train ../data/ko-universal-train.conll.shuffled --dev ../data/ko-universal-dev.conll --extrn emb100.ko --wembedding 100 --cembedding 100  --usechar --usejamo
```

### Parsing Commands
```
python src/parser.py --predict --outdir ../scratch/jamo --test ko-universal-test.conll --model ../scratch/jamo/model
```

### Reference
TODO