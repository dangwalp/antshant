# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import random
import numpy as np
import librosa
#import librosa.display
#import sounddevice as sd
#import matplotlib.pyplot as plt

from os import walk
from ruamel.yaml import YAML

from preprocess import load_wav


class Data:
    def __init__(self, path, target_inst, med_limit, sr, sec, vis=False):
        self.path = path
        self.target_inst = target_inst
        self.med_limit = med_limit
        self.file_tuples = self.stems_from_yaml()
        self.pwavs = self.prep_all_wavs(sec, sr)
        if vis:
            self.visualize_wavs(self.pwavs, sr)

    def prep_all_wavs(self, sec, sr):
        print("Preparing and caching audio files...")
        # Input: List of tuples [(target, [other1, other2, ...]), ...]
        #                        |....... song 1 ...............| song 2
        cache = []
        for med in self.file_tuples:
            print("Loading\t{}\nand other stems from the same directory.\n".format(med[0]))
            stems = []

            # Target
            target_stem, med_start = load_wav(med[0], sec, sr)
            
            # Other
            for stem in med[1]:
                other_stem, _ = load_wav(stem, sec, sr, med_st=med_start)
                stems.append(other_stem)
            mix_other = sum(stems)
            
            # All
            stems.append(target_stem)
            mix_all = sum(stems)

            # Medley name (for output files)
            med_name = ("".join(med[0].split('_')[:-2])).split('/')[-1]

            # Put everything together
            cache.append((mix_all, target_stem, mix_other, med_name))
            # Output: List of tuples [(mixed, src1, src2, name), ...]
        print("Done preparing audio files.")
        return cache

    def next_wavs(self, size=1):
        # Sample from preprocessed (already loaded) wav files and put into np.array
        rnd_medleys = random.sample(self.pwavs, size)
        mixed, src1, src2, medname = [], [], [], []
        for med in rnd_medleys:
            mixed.append(med[0])
            src1.append(med[1])
            src2.append(med[2])
            medname.append(med[3])
        return np.array(mixed), np.array(src1), np.array(src2), medname

    def stems_from_yaml(self):
        print("\nRetrieving {} audio files from yaml...".format(self.target_inst))
        yamlfiles = []
        for (root, dirs, files) in walk(self.path):
            yamlfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".yaml")
                and not f.startswith("._")])

        if len(yamlfiles) == 0:
            raise ValueError("No YAML files found in {}".format(self.path))

        file_tuples = []
        for y in yamlfiles:
            medley_stems = None
            target_stem = None
            other_stems = []
            if len(file_tuples) >= self.med_limit:
                print("Maximum medley limit of {} reached.".format(self.med_limit))
                break
            with open(y, 'r') as yf:
                yaml = YAML(typ='safe')
                #print("Current YAML: {}".format(yf))
                whole = yaml.load(yf)
                stem_dir = whole['stem_dir']
                stems = whole['stems']

                for st in stems.values():
                    stem_file = st["filename"]
                    stem_instrument = st["instrument"]
                    if stem_instrument == self.target_inst:
                        target_stem = "{}/{}/{}/{}".format(self.path,
                            "_".join(t for t in stem_dir.split('_')[:-1]),
                            stem_dir, stem_file)
                    else:
                        other_stems.append("{}/{}/{}/{}".format(self.path,
                            "_".join(t for t in stem_dir.split('_')[:-1]),
                            stem_dir, stem_file))
                # (Target instrument, [List of all other instruments of the same song])
                medley_stems = (target_stem, other_stems)
            if medley_stems:
                if medley_stems[0] and medley_stems[1]:
                    print("Found in\t{}".format(stem_dir))
                    file_tuples.append(medley_stems)
        print("")
        return file_tuples

    def visualize_wavs(self, wavs, sr):
        for med in wavs:
            mixed, src1, src2 = med
            plt.subplot(311)
            librosa.display.waveplot(librosa.to_mono(src1[:10*sr]), sr)
            plt.subplot(312)
            librosa.display.waveplot(librosa.to_mono(src2[:10*sr]), sr)
            plt.subplot(313)
            librosa.display.waveplot(librosa.to_mono(mixed[:10*sr]), sr)
            plt.show()
            input("Visualization done.")
