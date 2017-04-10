python src/parser.py --dynet-seed 123456789 --dynet-mem 1000 --outdir ../scratch/word
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/word-char --usechar
python src/parser.py --dynet-seed 123456789 --dynet-mem 3000 --outdir ../scratch/word-char-jamo --usechar --usejamo
python src/parser.py --dynet-seed 123456789 --dynet-mem 3000 --outdir ../scratch/word-char-jamo-highway --usechar --usejamo --highway

python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/word-emb --extrn ../data/emb100.ko
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/word-emb-char --extrn ../data/emb100.ko --usechar
python src/parser.py --dynet-seed 123456789 --dynet-mem 3000 --outdir ../scratch/word-emb-char-jamo --extrn ../data/emb100.ko --usechar --usejamo
python src/parser.py --dynet-seed 123456789 --dynet-mem 3000 --outdir ../scratch/word-emb-char-jamo-highway --extrn ../data/emb100.ko --usechar --usejamo --highway

python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/char --noword --usechar --cembedding 100
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/jamo --noword --usejamo --cembedding 100
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/char-jamo --noword --usechar --usejamo --cembedding 100

python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/char-highway --noword --usechar --cembedding 100 --highway
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/jamo-highway --noword --usejamo --cembedding 100 --highway
python src/parser.py --dynet-seed 123456789 --dynet-mem 2000 --outdir ../scratch/char-jamo-highway --noword --usechar --usejamo --cembedding 100 --highway


python src/parser.py --dynet-seed 123456789 --dynet-mem 4000 --outdir ../scratch/char200-highway --noword --usechar --cembedding 200 --highway
python src/parser.py --dynet-seed 123456789 --dynet-mem 4000 --outdir ../scratch/jamo200-highway --noword --usejamo --cembedding 200 --highway
python src/parser.py --dynet-seed 123456789 --dynet-mem 4000 --outdir ../scratch/char200-jamo200-highway --noword --usechar --usejamo --cembedding 200 --highway
python src/parser.py --dynet-seed 123456789 --dynet-mem 6000 --outdir ../scratch/word-emb-char100-jamo100-highway --extrn ../data/emb100.ko --usechar --usejamo --cembedding 100 --highway

python src/parser.py --predict --model ../scratch/word/model25 --outdir ../scratch/word

# Quick
python src/parser.py ../scratch/quick --train ../data/ko-universal-train.conll.shuffled.small --dev ../data/ko-universal-train.conll.shuffled.small --test ../data/ko-universal-train.conll.shuffled.small --epochs 3 --lstmdims 50
