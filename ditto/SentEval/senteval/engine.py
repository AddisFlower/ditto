# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLIEval
from senteval.trec import TRECEval
from senteval.sick import SICKEntailmentEval, SICKEval
from senteval.mrpc import MRPCEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval, SICKRelatednessEval, STSBenchmarkFinetune, STSBenchmarkEvalES, STSBenchmarkEvalFR, STSBenchmarkEvalIT, STSBenchmarkEvalPT
from senteval.sst import SSTEval
from senteval.rank import ImageCaptionRetrievalEval
from senteval.probing import *

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        # added STSBenchmarkES for checkpoint 2 testing
        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16',
                           'Length', 'WordContent', 'Depth', 'TopConstituents',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion', 'SICKRelatedness-finetune', 'STSBenchmark-finetune', 'STSBenchmark-fix',
                           'STSBenchmarkES', 'STSBenchmarkFR', 'STSBenchmarkIT', 'STSBenchmarkPT',
                           'STSBenchmark_gs', 'STSBenchmarkES_gs', 'STSBenchmarkFR_gs', 'STSBenchmarkIT_gs', 'STSBenchmarkPT_gs']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'CR':
            self.evaluation = CREval(tpath + '/downstream/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + '/downstream/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + '/downstream/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + '/downstream/SUBJ', seed=self.params.seed)
        elif name == 'SST2':
            self.evaluation = SSTEval(tpath + '/downstream/SST/binary', nclasses=2, seed=self.params.seed)
        elif name == 'SST5':
            self.evaluation = SSTEval(tpath + '/downstream/SST/fine', nclasses=5, seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + '/downstream/TREC', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + '/downstream/MRPC', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/downstream/SICK', seed=self.params.seed)
        # modified for performing grid search
        # elif name == 'STSBenchmark':
        #     self.evaluation = STSBenchmarkEval(tpath + '/downstream/STS/STSBenchmark', seed=self.params.seed)
        elif name in ['STSBenchmark', 'STSBenchmark_gs']:
            self.evaluation = STSBenchmarkEval(tpath + '/downstream/STS/STSBenchmark', seed=self.params.seed)
        # added for STSB_ES
        # modified for performing grid search
        elif name in ['STSBenchmarkES', 'STSBenchmarkES_gs']:
            self.evaluation = STSBenchmarkEvalES(tpath + '/downstream/STS/STSBenchmarkES', seed=self.params.seed)
        # added for STSB_FR
        # modified for performing grid search
        elif name in ['STSBenchmarkFR', 'STSBenchmarkFR_gs']:
            self.evaluation = STSBenchmarkEvalFR(tpath + '/downstream/STS/STSBenchmarkFR', seed=self.params.seed)
        # added for STSB_IT
        # modified for performing grid search
        elif name in ['STSBenchmarkIT', 'STSBenchmarkIT_gs']:
            self.evaluation = STSBenchmarkEvalIT(tpath + '/downstream/STS/STSBenchmarkIT', seed=self.params.seed)
        # added for STSB_PT
        # modified for performing grid search
        elif name in ['STSBenchmarkPT', 'STSBenchmarkPT_gs']:
            self.evaluation = STSBenchmarkEvalPT(tpath + '/downstream/STS/STSBenchmarkPT', seed=self.params.seed)
        elif name == 'STSBenchmark-fix':
            self.evaluation = STSBenchmarkEval(tpath + '/downstream/STS/STSBenchmark-fix', seed=self.params.seed)
        elif name == 'STSBenchmark-finetune':
            self.evaluation = STSBenchmarkFinetune(tpath + '/downstream/STS/STSBenchmark', seed=self.params.seed)
        elif name == 'SICKRelatedness-finetune':
            self.evaluation = SICKEval(tpath + '/downstream/SICK', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/downstream/SICK', seed=self.params.seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(tpath + '/downstream/SNLI', seed=self.params.seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/downstream/STS/' + fpath, seed=self.params.seed)
            # for testing
            # print(self.evaluation)
        elif name == 'ImageCaptionRetrieval':
            self.evaluation = ImageCaptionRetrievalEval(tpath + '/downstream/COCO', seed=self.params.seed)

        # Probing Tasks
        elif name == 'Length':
                self.evaluation = LengthEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'WordContent':
                self.evaluation = WordContentEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'Depth':
                self.evaluation = DepthEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'TopConstituents':
                self.evaluation = TopConstituentsEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'BigramShift':
                self.evaluation = BigramShiftEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'Tense':
                self.evaluation = TenseEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'SubjNumber':
                self.evaluation = SubjNumberEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'ObjNumber':
                self.evaluation = ObjNumberEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'OddManOut':
                self.evaluation = OddManOutEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'CoordinationInversion':
                self.evaluation = CoordinationInversionEval(tpath + '/probing', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        # added conditional for performing grid search on STSBenchmark dataset
        if name in ['STSBenchmark_gs', 'STSBenchmarkES_gs', 'STSBenchmarkFR_gs', 'STSBenchmarkIT_gs', 'STSBenchmarkPT_gs']:
            self.results = self.evaluation.run_stsb_gs(self.params, self.batcher)
        else:
            self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
