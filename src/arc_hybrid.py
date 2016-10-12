from pycnn import *
from utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np
from jamo import decompose
from math import sqrt
import os

FIRST_REPORT = True

def disable_first_report():
    global FIRST_REPORT
    FIRST_REPORT = False

class ArcHybridLSTM:
    def __init__(self, words, pos, rels, w2i, jamos, j2i, chars, c2i, options):
        self.model = Model()
        self.trainer = AdamTrainer(self.model)
        random.seed(1)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.oracle = options.oracle
        self.ldims = options.lstm_dims * 2
        self.lcdim = options.lcdim  # output dim of char LSTM
        self.cdims = options.cembedding_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.jamosCount = jamos
        # 0: unknown non-Hangul symbol, 1: unknown Jamo, 2: empty consonant
        self.jvocab = {jamo: ind+2 for jamo, ind in j2i.iteritems()}
        self.jamo_cache = {}
        self.charsCount = chars
        self.cvocab = {char: ind+1 for char, ind in c2i.iteritems()}
        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.window

        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)

        self.blstmFlag = options.blstmFlag
        self.bibiFlag = options.bibiFlag
        self.noword = options.noword
        self.usechar = options.usechar
        self.usejamo = options.usejamo
        self.pretrain = options.pretrain
        self.dist = options.dist
        self.outdir = options.output

        if self.usechar or self.usejamo:
            self.model.add_lookup_parameters("char-lookup-root", (1, self.lcdim))
            inputdim = self.cdims
            if self.usechar and self.usejamo: inputdim += self.cdims
            self.charBuilder = LSTMBuilder(1, inputdim, self.lcdim, self.model)

            if self.usechar:
                self.model.add_lookup_parameters("char-lookup",
                                                 (len(chars) + 1, self.cdims))
            if self.usejamo:
                self.model.add_lookup_parameters("jamo-lookup", (len(jamos) + 3,
                                                                 self.cdims))
                self.model.add_parameters("jamo-layer",
                                          (self.cdims, 3 * self.cdims))
                self.model.add_parameters("jamo-bias", (self.cdims))

        dims = self.pdims  # Input dimension for word-level LSTMs
        if not self.noword:                dims += self.wdims
        if self.usechar or self.usejamo:   dims += self.lcdim

        if self.bibiFlag:
            self.surfaceBuilders = [LSTMBuilder(1, dims, self.ldims * 0.5, self.model),
                                    LSTMBuilder(1, dims, self.ldims * 0.5, self.model)]
            self.bsurfaceBuilders = [LSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model),
                                     LSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model)]
        elif self.blstmFlag:
            if self.layers > 0:
                self.surfaceBuilders = [LSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model), LSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model)]
            else:
                self.surfaceBuilders = [SimpleRNNBuilder(1, dims, self.ldims * 0.5, self.model), LSTMBuilder(1, dims, self.ldims * 0.5, self.model)]

        self.hidden_units = options.hidden_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        vocab_size = len(words) if not self.noword else 0
        self.model.add_lookup_parameters("word-lookup", (vocab_size + 3, self.wdims))
        self.model.add_lookup_parameters("pos-lookup", (len(pos) + 3, self.pdims))
        self.model.add_lookup_parameters("rels-lookup", (len(rels), self.rdims))

        self.model.add_parameters("word-to-lstm", (self.ldims, self.wdims + self.pdims))
        self.model.add_parameters("word-to-lstm-bias", (self.ldims))
        self.model.add_parameters("lstm-to-lstm", (self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.model.add_parameters("lstm-to-lstm-bias", (self.ldims))

        self.model.add_parameters("hidden-layer", (self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.model.add_parameters("hidden-bias", (self.hidden_units))

        self.model.add_parameters("output-layer", (3, self.hidden_units))
        self.model.add_parameters("output-bias", (3))

        self.model.add_parameters("rhidden-layer", (self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.model.add_parameters("rhidden-bias", (self.hidden_units))

        self.model.add_parameters("routput-layer", (2 * (len(self.irels) + 0) + 1, self.hidden_units))
        self.model.add_parameters("routput-bias", (2 * (len(self.irels) + 0) + 1))

    def __evaluate(self, stack, buf, train):
        topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [self.empty] for i in xrange(self.k) ]
        topBuffer = [ buf.roots[i].lstms if len(buf) > i else [self.empty] for i in xrange(1) ]

        input = concatenate(list(chain(*(topStack + topBuffer))))

        routput = (self.routLayer * self.activation(self.rhidLayer * input + self.rhidBias) + self.routBias)

        output = (self.outLayer * self.activation(self.hidLayer * input + self.hidBias) + self.outBias)

        scrs, uscrs = routput.value(), output.value()

        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        uscrs2 = uscrs[2]
        if train:
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            ret = [ [ (rel, 0, scrs[1 + j * 2] + uscrs1, routput[1 + j * 2 ] + output1) for j, rel in enumerate(self.irels) ] if len(stack) > 0 and len(buf) > 0 else [],
                    [ (rel, 1, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2 ] + output2) for j, rel in enumerate(self.irels) ] if len(stack) > 1 else [],
                    [ (None, 2, scrs[0] + uscrs0, routput[0] + output0) ] if len(buf) > 0 else [] ]
        else:
            s1,r1 = max(zip(scrs[1::2],self.irels))
            s2,r2 = max(zip(scrs[2::2],self.irels))
            s1 += uscrs1
            s2 += uscrs2
            ret = [ [ (r1, 0, s1) ] if len(stack) > 0 and len(buf) > 0 else [],
                    [ (r2, 1, s2) ] if len(stack) > 1 else [],
                    [ (None, 2, scrs[0] + uscrs0) ] if len(buf) > 0 else [] ]
        return ret
        #return [ [ (rel, 0, scrs[1 + j * 2 + 0] + uscrs[1], routput[1 + j * 2 + 0] + output[1]) for j, rel in enumerate(self.irels) ] if len(stack) > 0 and len(buf) > 0 else [],
        #         [ (rel, 1, scrs[1 + j * 2 + 1] + uscrs[2], routput[1 + j * 2 + 1] + output[2]) for j, rel in enumerate(self.irels) ] if len(stack) > 1 else [],
        #         [ (None, 2, scrs[0] + uscrs[0], routput[0] + output[0]) ] if len(buf) > 0 else [] ]


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)

    def Init(self):
        self.word2lstm = parameter(self.model["word-to-lstm"])
        self.lstm2lstm = parameter(self.model["lstm-to-lstm"])

        self.word2lstmbias = parameter(self.model["word-to-lstm-bias"])
        self.lstm2lstmbias = parameter(self.model["lstm-to-lstm-bias"])

        self.hidLayer = parameter(self.model["hidden-layer"])
        self.outLayer = parameter(self.model["output-layer"])

        self.hidBias = parameter(self.model["hidden-bias"])
        self.outBias = parameter(self.model["output-bias"])

        self.rhidLayer = parameter(self.model["rhidden-layer"])
        self.routLayer = parameter(self.model["routput-layer"])

        self.rhidBias = parameter(self.model["rhidden-bias"])
        self.routBias = parameter(self.model["routput-bias"])

        paddingWordVec = lookup(self.model["word-lookup"], 1)
        paddingPosVec = lookup(self.model["pos-lookup"], 1) if self.pdims > 0 else None

        paddingVec = tanh(self.word2lstm * concatenate(filter(None, [paddingWordVec, paddingPosVec])) + self.word2lstmbias )
        self.empty = paddingVec if self.nnvecs == 1 else concatenate([paddingVec for _ in xrange(self.nnvecs)])

    def keepOrDropJamo(self, jamo, train):
        jamo_count = float(self.jamosCount.get(jamo, 0))
        dropFlag = not train or \
            (random.random() < (jamo_count/(0.25+jamo_count)))
        # 1: unknown Jamo
        jamo_index = int(self.jvocab.get(jamo, 1)) if dropFlag else 1
        return lookup(self.model["jamo-lookup"], jamo_index)

    def getJamoVec(self, char, train):
        if not char in self.jamo_cache:
            self.jamo_cache[char] = decompose(char)
        jamos = self.jamo_cache[char]

        if len(jamos) == 1:  # Non-Hangul (ex: @, Q)
            symbol = jamos[0]
            symbol_count = float(self.jamosCount.get(symbol, 0))
            dropFlag = not train or \
                (random.random() < (symbol_count/(0.25+symbol_count)))
            # 0: unknown symbol
            jamo_index = int(self.jvocab.get(symbol, 0)) if dropFlag else 0
            return lookup(self.model["jamo-lookup"], jamo_index)

        # Hangul character
        jamo1vec = self.keepOrDropJamo(jamos[0], train)
        jamo2vec = self.keepOrDropJamo(jamos[1], train)
        jamo3vec = self.keepOrDropJamo(jamos[2], train) if len(jamos) > 2 else \
            lookup(self.model["jamo-lookup"], 2)  # 2: empty consonant
        jamoinput = concatenate([ jamo1vec, jamo2vec, jamo3vec ])

        jamoLayer = parameter(self.model["jamo-layer"])
        jamoBias = parameter(self.model["jamo-bias"])
        jamovec = self.activation(jamoLayer * jamoinput + jamoBias)

        return jamovec

    def getCharVec(self, char, train):
        char_count = float(self.charsCount.get(char, 0))
        dropFlag = not train or \
            (random.random() < (char_count/(0.25+char_count)))
        char_index = int(self.cvocab.get(char, 0)) if dropFlag else 0
        return lookup(self.model["char-lookup"], char_index)

    def getCharacterEmbedding(self, word, train):
        if word == "*root*": return lookup(self.model["char-lookup-root"], 0)

        cforward  = self.charBuilder.initial_state()

        # Forward
        for char in unicode(word,"utf-8"):
            charvec = self.getCharVec(char, train) if self.usechar else None
            jamovec = self.getJamoVec(char, train) if self.usejamo else None
            cinput = concatenate(filter(None, [charvec, jamovec]))
            cforward = cforward.add_input(cinput)

        result = cforward.output()

        return result

    def getInitialWordEmbedding(self, word, pos, form, train):
        charvec = self.getCharacterEmbedding(word, train) \
            if self.usechar or self.usejamo else None  # 2 * self.cdims

        c = float(self.wordsCount.get(word, 0))
        dropFlag =  not train or (random.random() < (c/(0.25+c)))
        wordvec = None if self.noword else \
            lookup(self.model["word-lookup"], int(self.vocab.get(word, 0))
                   if dropFlag else 0)
        posvec = lookup(self.model["pos-lookup"], int(self.pos[pos])) \
            if self.pdims > 0 else None

        result = concatenate(filter(None, [charvec, wordvec, posvec]))

        return result

    def getWordEmbeddings(self, sentence, train):
        for root in sentence:
            root.ivec = self.getInitialWordEmbedding(root.norm, root.pos,
                                                     root.form, train)

            if FIRST_REPORT and root.norm != "*root*":
                print "+++++++++++++++++++++++"
                print "Sanity report: word \"{0}\" gets dimension {1} before " \
                    "LSTMs".format(root.norm, len(root.ivec.vec_value()))
                print "+++++++++++++++++++++++"
                disable_first_report()

        if self.blstmFlag:
            forward  = self.surfaceBuilders[0].initial_state()
            backward = self.surfaceBuilders[1].initial_state()

            for froot, rroot in zip(sentence, reversed(sentence)):
                forward = forward.add_input( froot.ivec )
                backward = backward.add_input( rroot.ivec )
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in sentence:
                root.vec = concatenate( [root.fvec, root.bvec] )

            if self.bibiFlag:
                bforward  = self.bsurfaceBuilders[0].initial_state()
                bbackward = self.bsurfaceBuilders[1].initial_state()

                for froot, rroot in zip(sentence, reversed(sentence)):
                    bforward = bforward.add_input( froot.vec )
                    bbackward = bbackward.add_input( rroot.vec )
                    froot.bfvec = bforward.output()
                    rroot.bbvec = bbackward.output()
                for root in sentence:
                    root.vec = concatenate( [root.bfvec, root.bbvec] )

        else:
            for root in sentence:
                root.ivec = (self.word2lstm * root.ivec) + self.word2lstmbias
                root.vec = tanh( root.ivec )

    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, False)):
                self.Init()

                sentence = sentence[1:] + [sentence[0]]
                self.getWordEmbeddings(sentence, False)
                stack = ParseForest([])
                buf = ParseForest(sentence)

                for root in sentence:
                    root.lstms = [root.vec for _ in xrange(self.nnvecs)]

                hoffset = 1 if self.headFlag else 0

                while len(buf) > 0 or len(stack) > 1 :
                    scores = self.__evaluate(stack, buf, False)
                    best = max(chain(*scores), key = itemgetter(2) )

                    if best[1] == 2:
                        stack.roots.append(buf.roots[0])
                        del buf.roots[0]

                    elif best[1] == 0:
                        child = stack.roots.pop()
                        parent = buf.roots[0]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 0
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                    elif best[1] == 1:
                        child = stack.roots.pop()
                        parent = stack.roots[-1]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 1
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                renew_cg()
                yield [sentence[-1]] + sentence[:-1]

    def Pretrain(self, external_embedding, num_epochs):
        with open(os.path.join(self.outdir, 'pretrain-initial'), 'w') as predf:
            for word in external_embedding:
                renew_cg()
                pred = self.getCharacterEmbedding(word, False).vec_value()
                predf.write(word)
                for v in pred: predf.write(' '+str(v))
                predf.write('\n')

        renew_cg()
        trainer = AdamTrainer(self.model)
        errs = []
        for epoch in xrange(num_epochs):
            print 'Pretraining epoch', epoch,
            total_loss = 0.0
            for word in external_embedding:
                gold = vecInput(self.wdims)
                gold.set(external_embedding[word])
                pred = self.getCharacterEmbedding(word, True)

                if self.dist == "l1":  # L1 norm
                    err = l1_distance(gold, pred)
                    #diff = gold - pred
                    #err = esum([emax([pick(diff,i), -pick(diff,i)]) for i in xrange(self.wdims)])

                elif self.dist == "l2":  # L2 norm
                    err = squared_distance(gold, pred)
                    #diff = gold - pred
                    #sqdiff = cwise_multiply(diff, diff)
                    #err = esum([pick(sqdiff,i) for i in xrange(self.wdims)])

                elif self.dist == "inf":  # Linf norm
                    diff = gold - pred
                    err = emax([emax([pick(diff,i), -pick(diff,i)]) for i in xrange(self.wdims)])

                elif self.dist == "cos":  # NEGATIVE cosine similarity
                    sqgold = cwise_multiply(gold, gold)
                    gold_norm = scalarInput(sqrt(esum([pick(sqgold,i) for i in xrange(self.wdims)]).value()))
                    gold_unit = cdiv(gold, concatenate([gold_norm for i in xrange(self.wdims)]))

                    sqpred = cwise_multiply(pred, pred)
                    pred_norm = scalarInput(sqrt(esum([pick(sqpred,i) for i in xrange(self.wdims)]).value()))
                    pred_unit = cdiv(pred, concatenate([pred_norm for i in xrange(self.wdims)]))

                    err = -dot_product(gold_unit, pred_unit)

                else:
                    print 'unknown dist:', self.dist
                    assert False

                errs.append(err)
                loss = err.scalar_value()
                total_loss += loss

                if len(errs) > 50:
                    eerrs = esum(errs)
                    eerrs.scalar_value()
                    eerrs.backward()
                    trainer.update()
                    errs = []
                    renew_cg()

            print "Loss: ", total_loss / len(external_embedding)
            total_loss = 0.0
            trainer.update_epoch()

        with open(os.path.join(self.outdir, 'pretrain-final'), 'w') as predf:
            for word in external_embedding:
                renew_cg()
                pred = self.getCharacterEmbedding(word, False).vec_value()
                predf.write(word)
                for v in pred: predf.write(' '+str(v))
                predf.write('\n')

    def Train(self, conll_path):
        mloss = 0.0
        errors = 0
        batch = 0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ltotal = 0
        ninf = -float('inf')

        hoffset = 1 if self.headFlag else 0

        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, True))

            ############ UNCOMMENT THIS WHEN YOU'RE USING UNSHUFFLED DATA! #####
            #random.shuffle(shuffledData)

            errs = []
            eeloss = 0.0

            self.Init()

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                sentence = sentence[1:] + [sentence[0]]
                self.getWordEmbeddings(sentence, True)
                stack = ParseForest([])
                buf = ParseForest(sentence)

                for root in sentence:
                    root.lstms = [root.vec for _ in xrange(self.nnvecs)]

                hoffset = 1 if self.headFlag else 0

                while len(buf) > 0 or len(stack) > 1 :
                    scores = self.__evaluate(stack, buf, True)
                    scores.append([(None, 3, ninf ,None)])

                    alpha = stack.roots[:-2] if len(stack) > 2 else []
                    s1 = [stack.roots[-2]] if len(stack) > 1 else []
                    s0 = [stack.roots[-1]] if len(stack) > 0 else []
                    b = [buf.roots[0]] if len(buf) > 0 else []
                    beta = buf.roots[1:] if len(buf) > 1 else []

                    left_cost  = ( len([h for h in s1 + beta if h.id == s0[0].parent_id]) +
                                   len([d for d in b + beta if d.parent_id == s0[0].id]) )  if len(scores[0]) > 0 else 1
                    right_cost = ( len([h for h in b + beta if h.id == s0[0].parent_id]) +
                                   len([d for d in b + beta if d.parent_id == s0[0].id]) )  if len(scores[1]) > 0 else 1
                    shift_cost = ( len([h for h in s1 + alpha if h.id == b[0].parent_id]) +
                                   len([d for d in s0 + s1 + alpha if d.parent_id == b[0].id]) )  if len(scores[2]) > 0 else 1
                    costs = (left_cost, right_cost, shift_cost, 1)

                    bestValid = max(( s for s in chain(*scores) if costs[s[1]] == 0 and ( s[1] == 2 or  s[0] == stack.roots[-1].relation ) ), key=itemgetter(2))
                    bestWrong = max(( s for s in chain(*scores) if costs[s[1]] != 0 or  ( s[1] != 2 and s[0] != stack.roots[-1].relation ) ), key=itemgetter(2))
                    best = bestValid if ( (not self.oracle) or (bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong

                    if best[1] == 2:
                        stack.roots.append(buf.roots[0])
                        del buf.roots[0]

                    elif best[1] == 0:
                        child = stack.roots.pop()
                        parent = buf.roots[0]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 0
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                    elif best[1] == 1:
                        child = stack.roots.pop()
                        parent = stack.roots[-1]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 1
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                    if bestValid[2] < bestWrong[2] + 1.0:
                        loss = bestWrong[3] - bestValid[3]
                        mloss += 1.0 + bestWrong[2] - bestValid[2]
                        eloss += 1.0 + bestWrong[2] - bestValid[2]
                        errs.append(loss)

                    if best[1] != 2 and (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                        lerrors += 1
                        if child.pred_parent_id != child.parent_id:
                            errors += 1
                            eerrors += 1

                    etotal += 1

                if len(errs) > 50: # or True:
                    #eerrs = ((esum(errs)) * (1.0/(float(len(errs)))))
                    eerrs = esum(errs)
                    scalar_loss = eerrs.scalar_value()
                    eerrs.backward()
                    self.trainer.update()
                    errs = []
                    lerrs = []

                    renew_cg()
                    self.Init()

        if len(errs) > 0:
            eerrs = (esum(errs)) # * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []

            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence
