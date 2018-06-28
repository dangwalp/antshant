# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import tensorflow as tf
import numpy as np
import os
import shutil
import argparse

from config import EvalConfig, ModelConfig
from data import Data
from mir_eval.separation import bss_eval_sources
from model import Model
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only
from preprocess import soft_time_freq_mask, to_wav, write_wav


def eval(data_path, instrument):
    # Model
    model = Model()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.Session(config=EvalConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, EvalConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(EvalConfig.GRAPH_PATH, sess.graph)

        data = Data("{}/{}".format(data_path, EvalConfig.DATA_PATH), instrument,
            EvalConfig.SECONDS)
        mixed_wav, src1_wav, src2_wav = data.next_wavs(EvalConfig.NUM_EVAL)

        mixed_spec = to_spectrogram(mixed_wav)
        mixed_mag = get_magnitude(mixed_spec)
        mixed_batch, padded_mixed_mag = model.spec_to_batch(mixed_mag)
        mixed_phase = get_phase(mixed_spec)

        assert (np.all(np.equal(model.batch_to_spec(mixed_batch, EvalConfig.NUM_EVAL),
            padded_mixed_mag)))

        (pred_src1_mag, pred_src2_mag) = sess.run(model(),
            feed_dict={model.x_mixed: mixed_batch})

        seq_len = mixed_phase.shape[-1]
        pred_src1_mag = model.batch_to_spec(pred_src1_mag, EvalConfig.NUM_EVAL)[:, :, :seq_len]
        pred_src2_mag = model.batch_to_spec(pred_src2_mag, EvalConfig.NUM_EVAL)[:, :, :seq_len]

        # Time-frequency masking
        mask_src1 = soft_time_freq_mask(pred_src1_mag, pred_src2_mag)
        # mask_src1 = hard_time_freq_mask(pred_src1_mag, pred_src2_mag)
        mask_src2 = 1. - mask_src1
        pred_src1_mag = mixed_mag * mask_src1
        pred_src2_mag = mixed_mag * mask_src2

        # (magnitude, phase) -> spectrogram -> wav
        if EvalConfig.GRIFFIN_LIM:
            pred_src1_wav = to_wav_mag_only(pred_src1_mag, init_phase=mixed_phase,
                num_iters=EvalConfig.GRIFFIN_LIM_ITER)
            pred_src2_wav = to_wav_mag_only(pred_src2_mag, init_phase=mixed_phase,
                num_iters=EvalConfig.GRIFFIN_LIM_ITER)
        else:
            pred_src1_wav = to_wav(pred_src1_mag, mixed_phase)
            pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)

        # Write the result
        tf.summary.audio('GT_mixed', mixed_wav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)
        tf.summary.audio('Pred_music', pred_src1_wav, ModelConfig.SR,
            max_outputs=EvalConfig.NUM_EVAL)
        tf.summary.audio('Pred_vocal', pred_src2_wav, ModelConfig.SR,
            max_outputs=EvalConfig.NUM_EVAL)

        if EvalConfig.EVAL_METRIC:
            # Compute BSS metrics
            gnsdr, gsir, gsar = bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav,
                pred_src2_wav)

            # Write the score of BSS metrics
            tf.summary.scalar('GNSDR_music', gnsdr[0])
            tf.summary.scalar('GSIR_music', gsir[0])
            tf.summary.scalar('GSAR_music', gsar[0])
            tf.summary.scalar('GNSDR_vocal', gnsdr[1])
            tf.summary.scalar('GSIR_vocal', gsir[1])
            tf.summary.scalar('GSAR_vocal', gsar[1])

        if EvalConfig.WRITE_RESULT:
            # Write the result
            for i in range(len(mixed_wav)):
                write_wav(mixed_wav[i], '{}/{}'.format(EvalConfig.RESULT_PATH,
                    'all_stems_mixed'))
                write_wav(pred_src1_wav[i], '{}/{}'.format(EvalConfig.RESULT_PATH,
                    'target_instrument'))
                write_wav(pred_src2_wav[i], '{}/{}'.format(EvalConfig.RESULT_PATH,
                    'other_stems_mixed'))

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=global_step.eval())

        writer.close()


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:, :len_cropped]
    src2_wav = src2_wav[:, :len_cropped]
    mixed_wav = mixed_wav[:, :len_cropped]
    gnsdr = gsir = gsar = np.zeros(2)
    total_len = 0
    for i in range(EvalConfig.NUM_EVAL):
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


def setup_path():
    if EvalConfig.RE_EVAL:
        if os.path.exists(EvalConfig.GRAPH_PATH):
            shutil.rmtree(EvalConfig.GRAPH_PATH)
        if os.path.exists(EvalConfig.RESULT_PATH):
            shutil.rmtree(EvalConfig.RESULT_PATH)

    if not os.path.exists(EvalConfig.RESULT_PATH):
        os.makedirs(EvalConfig.RESULT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", help="path to MedleyDB",
        default='/data', type=str, dest='dp')
    parser.add_argument("-i", "--instrument", help="target instrument",
        default='acoustic guitar', type=str, dest='inst')
    args = parser.parse_args()

    setup_path()
    eval(data_path=args.dp, instrument=args.inst)
