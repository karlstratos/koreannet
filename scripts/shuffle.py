import sys
import random

infile = open(sys.argv[1], 'r')
outfile = open(sys.argv[2], 'w')
sents = []
sent = []
for line in infile:
    if len(line.split()) > 0:
        sent.append(line)
    else:
        if sent:
            sents.append(sent)
            sent = []

random.shuffle(sents)

for sent in sents:
    for line in sent:
        outfile.write(line)
    outfile.write("\n")

infile.close()
outfile.close()
