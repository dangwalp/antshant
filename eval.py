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

from config import get_train_conf
from data import Data
from mir_eval.separation import bss_eval_sources
from model import Model
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only
from preprocess import soft_time_freq_mask, to_wav, write_wav
from utils import closest_power_of_two


def eval(model, data, sr, len_frame, num_wav, glim, glim_iter, ckpt_path,
    graph_path, result_path):
    len_hop = closest_power_of_two(len_frame / 4)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.Session() as sess:
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, ckpt_path)

        writer = tf.summary.FileWriter("{}/{}".format(graph_path, "eval"), sess.graph)

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
        if glim:
            pred_src1_wav = to_wav_mag_only(pred_src1_mag, mixed_phase, len_frame,
                len_hop, num_iters=glim_iter)
            pred_src2_wav = to_wav_mag_only(pred_src2_mag, mixed_phase, len_frame,
                len_hop, num_iters=glim_iter)
        else:
            pred_src1_wav = to_wav(pred_src1_mag, mixed_phase, len_hop)
            pred_src2_wav = to_wav(pred_src2_mag, mixed_phase, len_hop)

        # Write the result
        # [THIS LEADS TO A TYPE ERROR!]
        #tf.summary.audio('GT_mixed', mixed_wav, sr, max_outputs=num_wav)
        #tf.summary.audio('Pred_music', pred_src1_wav, sr, max_outputs=num_wav)
        #tf.summary.audio('Pred_vocal', pred_src2_wav, sr, max_outputs=num_wav)

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

        # Write the result
        for i in range(len(med_names)):
            write_wav(mixed_wav[i], '{}/{}-{}'.format(result_path, med_names[i],
                'all_stems_mixed'), sr)
            write_wav(pred_src1_wav[i], '{}/{}-{}'.format(result_path, med_names[i],
                'target_instrument'), sr)
            write_wav(pred_src2_wav[i], '{}/{}-{}'.format(result_path, med_names[i],
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
    parser.add_argument("-sec", "--seconds", help="length of snippet from audio",
        default=60, type=int, dest='sec')
    # Eval
    parser.add_argument("-nwav", "--num_wav", help="number of input files",
        default=5, type=int, dest='nwav')
    args = parser.parse_args()

    tconf = get_train_conf()

    data = Data("{}/{}".format(tconf['dp'], tconf['AUDIOFILES_PATH']), tconf['inst'],
        int(tconf['mdl']), int(tconf['sr']), args.sec, choose_eval=True)

    model = Model(sample_rate=int(tconf['sr']), len_frame=int(tconf['lfr']), seq_len=int(tconf['sql']),
        n_rnn_layer=int(tconf['layers']), hidden_size=int(tconf['hidden']))

    eval(model, data, sr=int(tconf['sr']), len_frame=int(tconf['lfr']), num_wav=args.nwav,
        glim=tconf['GRIFFIN_LIM'], glim_iter=int(tconf['GRIFFIN_LIM_ITER']), ckpt_path=tconf['CKPT_PATH'],
        graph_path=tconf['GRAPH_PATH'], result_path=tconf['RESULT_PATH'])

