# KoreanNet
__KoreanNet__ is a neural architecture for modeling the unique compositional orthography of the Korean language. It decomposes each character to extract underlying __jamo__ letters (basic phonetic units) from which character represenations are composed. For example, the embedding for the character `갔` is a function of `ㄱ`, `ㅏ`, and `ㅆ`. The decomposition can be performed deterministically and efficiently by simple Unicode manipulation: we use a robust implementation provided [here](https://github.com/JDongian/python-jamo).

This package plugs KoreanNet into the wonderful BiLSTM parser of [Kiperwasser and Goldberg (2016)](https://github.com/elikip/bist-parser). The jamo-based model achieves the high performance of character-based models while eschewing the need to store combinatorially many character types as lookup parameters. There is further improvement when jamos, characters, and words are used in conjunction. For details please refer to the paper.

### Prerequisites
- [Dynet for Python (version 1.1)](https://github.com/clab/dynet/tree/v1.1)
- [Korean treebank in the universal treebank v2](https://github.com/ryanmcd/uni-dep-tb)

### Training Commands
The code has disabled data shuffling for reproducibility in experiments. To enable shuffling, uncomment `random.shuffle(shuffledData)` in arc_hybrid.py. The Korean word embeddings were induced by running [CCA](http://karlstratos.com/publications/acl15cca.pdf) on a Wikipedia dump and are available [here](http://karlstratos.com/publications/emb100.ko.tar.gz).

###### Word only
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/word --train ../data/ko-universal-train.conll.shuffled --dev ../data/ko-universal-dev.conll --extrn ../data/emb100.ko --wembedding 100
```
###### Character only
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/char --train ../data/ko-universal-train.conll.shuffled --dev ../data/ko-universal-dev.conll  --cembedding 100  --usechar --noword
```
###### Jamo only
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/jamo --train ../data/ko-universal-train.conll.shuffled --dev ../data/ko-universal-dev.conll  --cembedding 100  --usejamo --noword
```
###### Word, character, and jamo
```
python src/parser.py --dynet-seed 123456789 --dynet-mem 3000 --outdir ../scratch/word-char-jamo --train ../data/ko-universal-train.conll.shuffled --dev ../data/ko-universal-dev.conll --extrn  ../data/emb100.ko --wembedding 100 --cembedding 100  --usechar --usejamo
```

### Parsing Commands
```
python src/parser.py --predict --outdir ../scratch/jamo --test  ../data/ko-universal-test.conll --model ../scratch/jamo/model
```

### Jamo Decomposition
If you want to just use the jamo decomposition for your task, the `decompose` function in jamo.py is the one you are looking for.

### Reference
 [__A Sub-Character Architecture for Korean Language Processing__](?).