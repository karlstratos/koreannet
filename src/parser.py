from optparse import OptionParser
from arc_hybrid import ArcHybridLSTM
from jamo import decompose
import pickle, utils, os, time, sys


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="../data/PTB_SD_3_3_0/train.conll")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="../data/PTB_SD_3_3_0/dev.conll")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="../data/PTB_SD_3_3_0/test.conll")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="barchybrid.model")
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=50)
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=50)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", default=30)
    parser.add_option("--pepochs", type="int", default=10)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=50)
    parser.add_option("--k", type="int", dest="window", default=3)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="relu")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=50)
    parser.add_option("--lcdim", type="int", dest="lcdim", default=50)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=7)
    parser.add_option("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=True)
    parser.add_option("--noword", action="store_true", default=False)
    parser.add_option("--usechar", action="store_true", default=False)
    parser.add_option("--usejamo", action="store_true", default=False)
    parser.add_option("--pretrain", action="store_true", default=False)
    parser.add_option("--fbchar", action="store_true", default=False)
    parser.add_option("--highway", action="store_true", default=False)
    parser.add_option("--dist", type="string", default="l2")
    parser.add_option("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_option("--userlmost", action="store_true", dest="rlFlag", default=False)
    parser.add_option("--userl", action="store_true", dest="rlMostFlag", default=False)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--dynet-mem", type="int", dest="dynet_mem", default=512)

    (options, args) = parser.parse_args()

    if not options.predictFlag:
        if not (options.rlFlag or options.rlMostFlag or options.headFlag):
            print 'You must use either --userlmost or --userl or --usehead (you can use multiple)'
            sys.exit()

        jamos, j2i, chars, c2i, words, w2i, pos, rels = utils.vocab(options.conll_train)

        print '----------------------------'
        print len(words), 'wtypes,', len(chars), 'ctypes,', len(jamos), 'jtypes'
        print 'Use word?', not options.noword
        print 'Use char?', options.usechar
        print 'Use jamo?', options.usejamo
        print 'word dim:', options.wembedding_dims
        print 'char dim:', options.cembedding_dims
        print 'pos dim:', options.pembedding_dims
        print 'lcdim:', options.lcdim
        if options.pretrain: print 'Pretraining loss:', options.dist
        print '----------------------------'

        external_embedding = {}
        if options.external_embedding is not None:
            with open(options.external_embedding,'r') as external_embedding_fp:
                external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            assert options.wembedding_dims ==  len(external_embedding.values()[0])
            print '{0} external embeddings'.format(len(external_embedding))
            num_new_words = 0
            num_new_chars = 0
            num_new_jamos = 0
            for word in external_embedding:  # expand word vocab
                if not word in words:
                    num_new_words += 1
                    words[word] = 1
                    new_w = len(w2i)
                    w2i[word] = new_w

                for char in unicode(word, "utf-8"):  # expand char vocab
                    if not char in chars:
                        num_new_chars += 1
                        chars[char] = 1
                        new_c = len(c2i)
                        c2i[char] = new_c

                    for jamo in decompose(char):  # expand jamo vocab
                        if not jamo in jamos:
                            num_new_jamos += 1
                            jamos[jamo] = 1
                            new_j = len(j2i)
                            j2i[jamo] = new_j
            print 'Have {0} new words, {1} new chars, {2} new jamos from pretrained embeddings'.format(num_new_words, num_new_chars, num_new_jamos)

        if not os.path.exists(options.output): os.makedirs(options.output)  # Make directory if needed

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((jamos, j2i, chars, c2i, words, w2i, pos,
                         rels, options), paramsfp)

        print 'Initializing blstm arc hybrid:'
        parser = ArcHybridLSTM(words, pos, rels, w2i, jamos, j2i, chars, c2i,
                               options)

        if external_embedding and not options.noword:
            print 'Initializing parameters with word embeddings'
            for word in external_embedding:
                w = parser.vocab[word]  # DON'T use w2i, use parser.vocab!
                parser.wlookup.init_row(w, external_embedding[word])

        if options.pretrain:
            assert options.usechar or options.usejamo
            assert options.lcdim == options.wembedding_dims
            parser.Pretrain(external_embedding, options.pepochs)

        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.Train(options.conll_train)
            devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch+1) + '.conll')
            utils.write_conll(devpath, parser.Predict(options.conll_dev))
            os.system('perl src/utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath  + ' > ' + devpath + '.txt &')
            print 'Finished predicting dev'
            parser.Save(os.path.join(options.output, "model" + str(epoch+1)))
    else:
        with open(options.params, 'r') as paramsfp:
            jamos, j2i, chars, c2i, words, w2i, pos, \
                rels, stored_opt = pickle.load(paramsfp)

        print '# words: ', len(w2i)
        print '# chars: ', len(c2i)
        print '# jamos: ', len(j2i)
        parser = ArcHybridLSTM(words, pos, rels, w2i, jamos, j2i, chars, c2i,
                               stored_opt)
        parser.Load(options.model)
        tespath = os.path.join(options.output, 'test_pred.conll')
        ts = time.time()
        pred = list(parser.Predict(options.conll_test))
        te = time.time()
        utils.write_conll(tespath, pred)
        os.system('perl src/utils/eval.pl -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.txt &')
        print 'Finished predicting test',te-ts
