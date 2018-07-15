import os
import csv


class GeneralConfig:
    def __init__(self, case):
        self.AUDIOFILES_PATH = 'MedleyDB/Audio'
        self.CKPT_PATH = 'checkpoints/' + case
        self.GRAPH_PATH = 'graphs/' + case
        self.CKPT_STEP = 100
        self.GRIFFIN_LIM = False
        self.GRIFFIN_LIM_ITER = 1000
        self.RESULT_PATH = 'results/' + case
        print("Created config: {}".format(case))


def get_train_conf():
    latest_result = sorted(os.listdir('checkpoints'))[-1]
    try:
        with open("configs/{}.log".format(latest_result, 'r')) as cfile:
            tconf = dict(csv.reader(cfile, delimiter='\t'))
            return tconf
    except FileNotFoundError:
        print("Run train.py before evaluating.")
