# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''


class GeneralConfig:
    def __init__(self, case):
        self.AUDIOFILES_PATH = 'MedleyDB/Audio'
        self.CKPT_PATH = 'checkpoints/' + case
        self.GRAPH_PATH = 'graphs/' + case
        self.CKPT_STEP = 100
        self.GRIFFIN_LIM = False
        self.GRIFFIN_LIM_ITER = 1000
        self.RE_EVAL = True
        self.EVAL_METRIC = True
        self.WRITE_RESULT = True
        self.RESULT_PATH = 'results/' + case
        print("Created config: {}".format(case))
