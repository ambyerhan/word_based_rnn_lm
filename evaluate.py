from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

import model
from data_reader import load_data, DataReader

from train import FILE_NAME_LIST

flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir',   'cv',     'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')
flags.DEFINE_float  ('alpha',           1.0,                            'length penelaty factor')

# optimization
flags.DEFINE_integer('num_unroll_steps',    35,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          20,   'number of sequences to train on in parallel')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


def main(_):
    ''' Loads trained model and evaluates it on test split '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + ".index"):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length, flist = FILE_NAME_LIST[2], eos=FLAGS.EOS)

    test_reader = DataReader(word_tensors[0], FLAGS.batch_size, FLAGS.num_unroll_steps)

    print('initialized test dataset reader')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build inference graph '''
        with tf.variable_scope("Model"):
            m = model.inference_graph(
                    word_vocab_size=word_vocab.size,
                    word_embed_size=FLAGS.char_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    num_unroll_steps=FLAGS.num_unroll_steps,
                    dropout=0)
            m.update(model.score_graph(m.logits, FLAGS.batch_size, FLAGS.num_unroll_steps, FLAGS.alpha))

            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)
        print('Loaded model from', FLAGS.load_model, 'saved at global step', global_step.eval())

        ''' training starts here '''
        count = 0
        avg_loss = 0
        start_time = time.time()
        rnn_state = session.run(m.initial_rnn_state)
        for x, y in test_reader.iter():
            count += 1
            loss = session.run(m.loss, {
                m.input  : x,
                m.targets: y,
                m.initial_rnn_state: rnn_state
            })

            avg_loss += loss

        avg_loss /= count
        time_elapsed = time.time() - start_time

        print("test loss = %6.8f, perplexity = %6.8f" % (avg_loss, np.exp(avg_loss)))
        print("test samples:", count*FLAGS.batch_size, "time elapsed:", time_elapsed, "time per one batch:", time_elapsed/count)


if __name__ == "__main__":
    tf.app.run()
