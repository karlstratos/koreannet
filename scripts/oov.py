import os
import sys
sys.path.insert(0,'src')
import utils
import jamo as jpack

jamos_train, j2i_train, chars_train, c2i_train, words_train, w2i_train, pos_train, rels_train = utils.vocab(sys.argv[1])
jamos_dev, j2i_dev, chars_dev, c2i_dev, words_dev, w2i_dev, pos_dev, rels_dev = utils.vocab(sys.argv[2])

oov_word = 0
for word in words_dev:
    if not word in words_train:
        oov_word += 1
print 'OOV word: ', oov_word, ' / ', len(words_dev), ' ', float(oov_word) / len(words_dev) * 100

hangul_chars_train = {}
for char in chars_train:
    if len(jpack.decompose(char)) > 1:
        hangul_chars_train[char] = True
hangul_chars_dev = {}
for char in chars_dev:
    if len(jpack.decompose(char)) > 1:
        hangul_chars_dev[char] = True

oov_char = 0
for char in hangul_chars_dev:
    if not char in hangul_chars_train:
        oov_char += 1
print 'OOV char: ', oov_char, ' / ', len(hangul_chars_dev), ' ', float(oov_char) / len(hangul_chars_dev) * 100

hangul_jamos_train = {}
for jamo in jamos_train:
    if jpack.is_jamo(jamo):
        hangul_jamos_train[jamo] = True
hangul_jamos_dev = {}
for jamo in jamos_dev:
    if jpack.is_jamo(jamo):
        hangul_jamos_dev[jamo] = True

oov_jamo = 0
for jamo in hangul_jamos_dev:
    if not jamo in hangul_jamos_train:
        oov_jamo += 1
print 'OOV jamo: ', oov_jamo, ' / ', len(hangul_jamos_dev), ' ', float(oov_jamo) / len(hangul_jamos_dev) * 100
