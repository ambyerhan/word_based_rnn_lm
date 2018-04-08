from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

import model
from data_reader import load_data, DataReader


#FILE_NAME_LIST = ['train.txt', 'valid.txt', 'test.txt']
FILE_NAME_LIST = ['_train.txt.prepro', '_valid.txt.prepro', '_test.txt.prepro']

flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir',   'cv',     'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('word_embed_size', 650,                             'dimensionality of character embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       1.0,  'starting learning rate')
flags.DEFINE_float  ('decay_when',          1.0,  'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps',    8,   'number of timesteps to unroll for') # 35
flags.DEFINE_integer('batch_size',          2,   'number of sequences to train on in parallel') # 20
flags.DEFINE_integer('max_epochs',          25,   'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_integer('print_every',    5,    'how often to print current loss')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
flags.DEFINE_bool   ('save',           False,  'save the model or not')

FLAGS = flags.FLAGS


def main(_):
    ''' Trains model from data '''
    min = [1000, 1000, 1000, 1000] # [t_loss, t_ppl, v_loss, v_ppl]
    total_time = 0.

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    word_vocab, \
    char_vocab, \
    word_tensors, \
    char_tensors, \
    max_word_length = load_data(FLAGS.data_dir, FLAGS.max_word_length, flist = FILE_NAME_LIST, eos=FLAGS.EOS)

    train_reader = DataReader(word_tensors[FILE_NAME_LIST[0]], FLAGS.batch_size, FLAGS.num_unroll_steps)

    valid_reader = DataReader(word_tensors[FILE_NAME_LIST[1]], FLAGS.batch_size, FLAGS.num_unroll_steps)

    test_reader  = DataReader(word_tensors[FILE_NAME_LIST[2]], FLAGS.batch_size, FLAGS.num_unroll_steps)

    print('initialized all dataset readers')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build training graph '''
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = model.inference_graph(
                    word_vocab_size=word_vocab.size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    num_unroll_steps=FLAGS.num_unroll_steps,
                    dropout=FLAGS.dropout)
            train_model.update(model.loss_graph(train_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

            # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
            # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
            # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and max_grad_norm appropriately.
            # Thus, scaling gradients so that this trainer is exactly compatible with the original
            train_model.update(model.training_graph(train_model.loss * FLAGS.num_unroll_steps,
                    FLAGS.learning_rate, FLAGS.max_grad_norm))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep=5)

        ''' build graph for validation and testing (shares parameters with the training graph!) '''
        with tf.variable_scope("Model", reuse=True):
            valid_model = model.inference_graph(
                    word_vocab_size=word_vocab.size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    num_unroll_steps=FLAGS.num_unroll_steps,
                    dropout=0.0)
            valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

        if FLAGS.load_model:
            saver.restore(session, FLAGS.load_model)
            print('Loaded model from', FLAGS.load_model, 'saved at global step', train_model.global_step.eval())
        else:
            tf.global_variables_initializer().run()
            session.run(train_model.clear_char_embedding_padding)
            print('Created and initialized fresh model. Size:', model.model_size())

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=session.graph)

        ''' take learning rate from CLI, not from saved graph '''
        session.run(tf.assign(train_model.learning_rate, FLAGS.learning_rate))


        print("=" * 89)
        print("=" * 89)
        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0
        pi = 1 # 0 is for sum of grad_sses
        for v_name in list(all_weights): # sorted()
            v = all_weights[v_name]
            v_size = int(np.prod(np.array(v.shape.as_list())))
            print("%02d-Weight   %s\tshape   %s\ttsize    %d" % (pi, v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size))
            total_size += v_size
            pi += 1
        print("Total size %d, %.3fMiB" % (total_size, (total_size * 4) / (1024 * 1024)))
        print("-" * 89)


        ''' training starts here '''
        best_valid_loss = None
        rnn_state = session.run(train_model.initial_rnn_state)
        for epoch in range(1, FLAGS.max_epochs + 1):

            epoch_start_time = time.time()
            avg_train_loss = 0.0
            count = 0
            for x, y in train_reader.iter():
                count += 1
                start_time = time.time()

                loss, _, rnn_state, gradient_norm, step, _ = session.run([
                    train_model.loss,
                    train_model.train_op,
                    train_model.final_rnn_state,
                    train_model.global_norm,
                    train_model.global_step,
                    train_model.clear_char_embedding_padding
                ], {
                    train_model.input  : x,
                    train_model.targets: y,
                    train_model.initial_rnn_state: rnn_state
                })

                avg_train_loss += 0.05 * (loss - avg_train_loss)

                time_elapsed = time.time() - start_time

                if count % FLAGS.print_every == 0:
                    cur_lr = session.run(train_model.learning_rate)
                    print('%6d: -%d- [%5d/%5d], train_loss/ppl = %6.8f/%6.7f batch/secs = %.1fb/s, cur_lr = %2.5f, grad.norm=%6.8f' % (step,
                                                            epoch, count,
                                                            train_reader.length,
                                                            loss, np.exp(loss),
                                                            FLAGS.print_every / time_elapsed,
                                                            cur_lr,
                                                            gradient_norm))

            print('Epoch training time:', time.time()-epoch_start_time)
            total_time += (time.time() - epoch_start_time)

            # epoch done: time to evaluate
            avg_valid_loss = 0.0
            count = 0
            rnn_state = session.run(valid_model.initial_rnn_state)
            for x, y in valid_reader.iter():
                count += 1
                start_time = time.time()

                loss, rnn_state = session.run([
                    valid_model.loss,
                    valid_model.final_rnn_state
                ], {
                    valid_model.input  : x,
                    valid_model.targets: y,
                    valid_model.initial_rnn_state: rnn_state,
                })

                if count % FLAGS.print_every == 0:
                    print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                avg_valid_loss += loss / valid_reader.length

            print("at the end of epoch:", epoch)
            print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
            print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))
            if min[2] > avg_valid_loss:
                min[0] = avg_train_loss
                min[1] = np.exp(avg_train_loss)
                min[2] = avg_valid_loss
                min[3] = np.exp(avg_valid_loss)

            save_as = '%s/epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, avg_valid_loss)
            saver.save(session, save_as)
            print('Saved model', save_as)

            ''' write out summary events '''
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)
            ])
            summary_writer.add_summary(summary, step)

            ''' decide if need to decay learning rate '''
            if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                print('validation perplexity did not improve enough, decay learning rate')
                current_learning_rate = session.run(train_model.learning_rate)
                print('learning rate was:', current_learning_rate)
                current_learning_rate *= FLAGS.learning_rate_decay
                if current_learning_rate < 1.e-5:
                    print('learning rate too small - stopping now')
                    break

                session.run(train_model.learning_rate.assign(current_learning_rate))
                print('new learning rate is:', current_learning_rate)
            else:
                best_valid_loss = avg_valid_loss

        ''' test on the test set '''
        ave_test_loss = 0.
        trnn_state = session.run(valid_model.initial_rnn_state)
        for x, y in test_reader.iter():
            loss, trnn_state = session.run(
                [
                    valid_model.loss,
                    valid_model.final_rnn_state
                ],
                {
                    valid_model.input: x,
                    valid_model.targets: y,

                    valid_model.initial_rnn_state: trnn_state
                }
            )
            disp_loss = loss
            ave_test_loss += disp_loss / test_reader.length

        print("=" * 89)
        print("=" * 89)
        print("Total training time(not included the valid time): %f" % total_time)
        print("The best result:")
        print("train loss = %.3f, ppl = %.4f" % (min[0], min[1]))
        print("valid loss = %.3f, ppl = %.4f" % (min[2], min[3]))
        print("test  loss = %.3f, ppl = %.4f" % (ave_test_loss, np.exp(ave_test_loss)))
        print("=" * 89)


if __name__ == "__main__":
    tf.app.run()
