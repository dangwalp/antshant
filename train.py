# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import tensorflow as tf
import os
import argparse
import datetime

from config import GeneralConfig
from data import Data
from model import Model
from preprocess import to_spectrogram, get_magnitude
from utils import Diff, closest_power_of_two


def train(model, data, Config, lr, eps, num_wav, len_frame):
    len_hop = closest_power_of_two(len_frame / 4)

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn,
        global_step=global_step)

    summary_op = summaries(model, loss_fn)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if not os.path.exists(Config.CKPT_PATH):
            os.makedirs(Config.CKPT_PATH)
        model.load_state(sess, Config.CKPT_PATH)

        writer = tf.summary.FileWriter("{}/{}".format(Config.GRAPH_PATH, "train"), sess.graph)

        print("Starting training...")

        loss = Diff()
        for step in range(global_step.eval(), eps):
            mixed_wav, src1_wav, src2_wav, _ = data.next_wavs(num_wav)

            mixed_spec = to_spectrogram(mixed_wav, len_frame, len_hop)
            mixed_mag = get_magnitude(mixed_spec)

            src1_spec = to_spectrogram(src1_wav, len_frame, len_hop)
            src2_spec = to_spectrogram(src2_wav, len_frame, len_hop)
            src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)

            src1_batch, _ = model.spec_to_batch(src1_mag)
            src2_batch, _ = model.spec_to_batch(src2_mag)
            mixed_batch, _ = model.spec_to_batch(mixed_mag)

            l, _, summary = sess.run([loss_fn, optimizer, summary_op],
                                     feed_dict={model.x_mixed: mixed_batch,
                                                model.y_src1: src1_batch,
                                                model.y_src2: src2_batch})

            loss.update(l)
            print('step-{}\td_loss={:2.2f}\tloss={}'.format(step, loss.diff * 100,
                loss.value))

            writer.add_summary(summary, global_step=step)

            # Save state
            if step % Config.CKPT_STEP == 0:
                print("Saved checkpoint.")
                tf.train.Saver().save(sess, Config.CKPT_PATH + '/checkpoint',
                    global_step=step)

        writer.close()


def summaries(model, loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss, v))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src1)
    return tf.summary.merge_all()


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
        default=30, type=int, dest='sec')

    # Model
    parser.add_argument("-lay", "--layers", help="number of RNN layers",
        default=3, type=int, dest='layers')
    parser.add_argument("-hid", "--hidden", help="hidden size",
        default=256, type=int, dest='hidden')

    # Training
    parser.add_argument("-lr", "--learning_rate", help="learning rate",
        default=0.0001, type=float, dest='lr')
    parser.add_argument("-eps", "--epochs", help="number of steps",
        default=100000, type=int, dest='eps')
    parser.add_argument("-nwav", "--num_wav", help="number of input files",
        default=1, type=int, dest='nwav')

    args = parser.parse_args()

    # Other config
    case = "{}_{}".format((args.inst).replace(' ', '-'),
        str(datetime.datetime.now()).split('.')[0].replace(' ', '-'))
    conf = GeneralConfig(case)
    # Writing config log
    config_dir = 'configs'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    with open('{}/{}.log'.format(config_dir, case), 'w') as cfile:
        for arg in sorted(vars(args)):
            print(arg, getattr(args, arg))
            cfile.write("{}\t{}\n".format(arg, getattr(args, arg)))
        for k,v in conf.__dict__.items():
            print(k, v)
            cfile.write("{}\t{}\n".format(k,v))
        print("")

    # Create dataset object
    data = Data("{}/{}".format(args.dp, conf.AUDIOFILES_PATH), args.inst,
        args.mdl, args.sr, args.sec)

    # Create model object
    model = Model(sample_rate=args.sr, len_frame=args.lfr, seq_len=args.sql,
        n_rnn_layer=args.layers, hidden_size=args.hidden)

    # Start training
    train(model, data, conf, lr=args.lr, eps=args.eps, num_wav=args.nwav,
        len_frame=args.lfr)
