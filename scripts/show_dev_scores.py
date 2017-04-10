import os
import re
import sys

epoch_filename = {}

for root, dirs, filenames in os.walk(sys.argv[1]):
    for filename in filenames:
        if "dev" == filename[:3] and "txt" == filename[-3:]:
            epoch = int(re.findall(r'\d+', filename)[0])
            epoch_filename[epoch] = os.path.join(root, filename)

max_LAS = 0.0
bets_epoch = 0
for epoch in sorted(epoch_filename):
    with open(epoch_filename[epoch], 'r') as infile:
        lines = infile.readlines()
        UAS = lines[1].split()[-2]
        LAS = lines[0].split()[-2]
        print str(epoch) + '\t' + UAS+'\t'+LAS

        LAS = float(LAS)
        if LAS > max_LAS:
            max_LAS = LAS
            best_epoch = epoch

print 'max LAS', max_LAS, 'at epoch', best_epoch
