# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import yaml
import random
from os import walk
from config import ModelConfig
from preprocess import load_wav, mix_stems


class Data:
    def __init__(self, path, target_inst):
        self.path = path
        self.target_inst = target_inst
        self.file_tuples = self.stems_from_yaml()

    def next_wavs(self, sec, size=1):
        rnd_medley = random.sample(self.file_tuples, size)[0]

        other = []
        for f in rnd_medley[1]:
            w = load_wav(f, sec, ModelConfig.SR)
            other.append(w)
        src2 = mix_stems(other)

        src1 = load_wav(rnd_medley[0], sec, ModelConfig.SR) # target
        all_stems = other.append(src1)
        
        mixed = mix_stems(all_stems)

        return mixed, src1, src2

    def stems_from_yaml(self):
        print("Retrieving audiofiles from yaml...")
        print("Target stems: \n")
        yamlfiles = []
        for (root, dirs, files) in walk(self.path):
            yamlfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".yaml")])

        file_tuples = []
        for y in yamlfiles:
            with open(y, 'r') as yf:
                whole = yaml.load(yf)
                stem_dir = whole['stem_dir']
                stems = whole['stems']

                other_stems = []
                for st in stems.values():
                    stem_file = st["filename"]
                    stem_instrument = st["instrument"]

                    if stem_instrument == self.target_inst:
                        print(stem_file, stem_instrument)
                        target_stem = "{}/{}/{}/{}".format(self.path,
                            "_".join(t for t in stem_dir.split('_')[:-1]),
                            stem_dir, stem_file)
                    else:
                        other_stems.append("{}/{}/{}/{}".format(self.path,
                            "_".join(t for t in stem_dir.split('_')[:-1]),
                            stem_dir, stem_file))
                # (Target instrument, [List of all other instruments of the same song])
                medley_stems = (target_stem, other_stems)
            file_tuples.append(medley_stems)

        print("")
        return file_tuples
