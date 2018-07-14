# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import tensorflow as tf
import numpy as np
import os
import argparse
import datetime

from config import GeneralConfig
from data import Data
from mir_eval.separation import bss_eval_sources
from model import Model
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only
from preprocess import soft_time_freq_mask, to_wav, write_wav
from utils import closest_power_of_two


def eval(model, data, Config, sr, len_frame, num_wav):
    len_hop = closest_power_of_two(len_frame / 4)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.Session() as sess:
        if not os.path.exists(Config.RESULT_PATH):
            os.makedirs(Config.RESULT_PATH)

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, Config.CKPT_PATH)

        writer = tf.summary.FileWriter("{}/{}".format(Config.GRAPH_PATH, "eval"), sess.graph)

        mixed_wav, src1_wav, src2_wav, med_names = data.next_wavs(num_wav)

        mixed_spec = to_spectrogram(mixed_wav, len_frame, len_hop)
        mixed_mag = get_magnitude(mixed_spec)
        mixed_batch, padded_mixed_mag = model.spec_to_batch(mixed_mag)
        mixed_phase = get_phase(mixed_spec)

        assert (np.all(np.equal(model.batch_to_spec(mixed_batch, num_wav),
            padded_mixed_mag)))

        (pred_src1_mag, pred_src2_mag) = sess.run(model(),
            feed_dict={model.x_mixed: mixed_batch})

        seq_len = mixed_phase.shape[-1]
        pred_src1_mag = model.batch_to_spec(pred_src1_mag, num_wav)[:, :, :seq_len]
        pred_src2_mag = model.batch_to_spec(pred_src2_mag, num_wav)[:, :, :seq_len]

        # Time-frequency masking
        mask_src1 = soft_time_freq_mask(pred_src1_mag, pred_src2_mag)
        # mask_src1 = hard_time_freq_mask(pred_src1_mag, pred_src2_mag)
        mask_src2 = 1. - mask_src1
        pred_src1_mag = mixed_mag * mask_src1
        pred_src2_mag = mixed_mag * mask_src2

        # (magnitude, phase) -> spectrogram -> wav
        if Config.GRIFFIN_LIM:
            pred_src1_wav = to_wav_mag_only(pred_src1_mag, mixed_phase, len_frame,
                len_hop, num_iters=Config.GRIFFIN_LIM_ITER)
            pred_src2_wav = to_wav_mag_only(pred_src2_mag, mixed_phase, len_frame,
                len_hop, num_iters=Config.GRIFFIN_LIM_ITER)
        else:
            pred_src1_wav = to_wav(pred_src1_mag, mixed_phase, len_hop)
            pred_src2_wav = to_wav(pred_src2_mag, mixed_phase, len_hop)

        # Write the result
        tf.summary.audio('GT_mixed', mixed_wav, sr, max_outputs=num_wav)
        tf.summary.audio('Pred_music', pred_src1_wav, sr, max_outputs=num_wav)
        tf.summary.audio('Pred_vocal', pred_src2_wav, sr, max_outputs=num_wav)

        if Config.EVAL_METRIC:
            # Compute BSS metrics
            gnsdr, gsir, gsar = bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav,
                pred_src2_wav, num_wav)

            # Write the score of BSS metrics
            tf.summary.scalar('GNSDR_music', gnsdr[0])
            tf.summary.scalar('GSIR_music', gsir[0])
            tf.summary.scalar('GSAR_music', gsar[0])
            tf.summary.scalar('GNSDR_vocal', gnsdr[1])
            tf.summary.scalar('GSIR_vocal', gsir[1])
            tf.summary.scalar('GSAR_vocal', gsar[1])

        if Config.WRITE_RESULT:
            # Write the result
            for i in range(len(med_names)):
                write_wav(mixed_wav[i], '{}/{}-{}'.format(Config.RESULT_PATH, med_names[i],
                    'all_stems_mixed'), sr)
                write_wav(pred_src1_wav[i], '{}/{}-{}'.format(Config.RESULT_PATH, med_names[i],
                    'target_instrument'), sr)
                write_wav(pred_src2_wav[i], '{}/{}-{}'.format(Config.RESULT_PATH, med_names[i],
                    'other_stems_mixed'), sr)

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=global_step.eval())

        writer.close()


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav, num_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:, :len_cropped]
    src2_wav = src2_wav[:, :len_cropped]
    mixed_wav = mixed_wav[:, :len_cropped]
    gnsdr = gsir = gsar = np.zeros(2)
    total_len = 0
    for i in range(num_wav):
        sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav[i], src2_wav[i]]),
                                            np.array([pred_src1_wav[i], pred_src2_wav[i]]), False)
        sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav[i], src2_wav[i]]),
                                              np.array([mixed_wav[i], mixed_wav[i]]), False)
        nsdr = sdr - sdr_mixed
        gnsdr += len_cropped * nsdr
        gsir += len_cropped * sir
        gsar += len_cropped * sar
        total_len += len_cropped
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    return gnsdr, gsir, gsar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-dp", "--data_path", help="path to MedleyDB root",
        default='/data', type=str, dest='dp')
    parser.add_argument("-inst", "--instrument", help="target instrument",
        default='acoustic guitar', type=str, dest='inst')
    parser.add_argument("-lfr", "--l_frame", help="l_frame",
        default=1024, type=int, dest='lfr')
    parser.add_argument("-sql", "--seq_len", help="sequence length",
        default=4, type=int, dest='sql')
    parser.add_argument("-mdl", "--med_limit", help="max medleys for training",
        default=5, type=int, dest='mdl')
    parser.add_argument("-sr", "--sample_rate",
        help="sample rate (MedleyDB original: 44100)", default=16000, type=int,
        dest='sr')
    parser.add_argument("-sec", "--seconds", help="length of snippet from audio",
        default=60, type=int, dest='sec')

    # Model
    parser.add_argument("-lay", "--layers", help="number of RNN layers",
        default=3, type=int, dest='layers')
    parser.add_argument("-hid", "--hidden", help="hidden size",
        default=256, type=int, dest='hidden')

    # Eval
    parser.add_argument("-nwav", "--num_wav", help="number of input files",
        default=1, type=int, dest='nwav')

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    conf = GeneralConfig("{}_{}".format((args.inst).replace(' ', '-'),
        str(datetime.datetime.now()).split('.')[0].replace(' ', '-')))

    data = Data("{}/{}".format(args.dp, conf.AUDIOFILES_PATH), args.inst,
        args.mdl, args.sr, args.sec)

    model = Model(sample_rate=args.sr, len_frame=args.lfr, seq_len=args.sql,
        n_rnn_layer=args.layers, hidden_size=args.hidden)

    eval(model, data, conf, sr=args.sr, len_frame=args.lfr, num_wav=args.nwav)
