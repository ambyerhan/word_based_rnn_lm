from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import pickle
import numpy as np


class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_word_length, flist, eos='+'):

    word_vocab = Vocab()

    word_tokens = collections.defaultdict(list)

    for fname in flist:
        print('reading', fname)
        with codecs.open(os.path.join(data_dir, fname), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()

                for word in line.split():

                    word_tokens[fname].append(word_vocab.feed(word))

                if eos:
                    word_tokens[fname].append(word_vocab.feed(eos))

    print()
    print('size of word vocabulary:', word_vocab.size)
    for fn in flist:
        print('number of tokens in %s: %d' % (fn, len(word_tokens[fn])))

    # now we know the sizes, create tensors
    word_tensors = {}
    for fname in flist:

        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)

    return word_vocab, None, word_tensors, None, None


class DataReader:

    def __init__(self, word_tensor, batch_size, num_unroll_steps):

        length = word_tensor.shape[0]

        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        word_tensor = word_tensor[:reduced_length]

        xdata = np.zeros_like(word_tensor)
        xdata[:] = word_tensor[:].copy()

        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()

        x_batches = xdata.reshape([batch_size, -1, num_unroll_steps])
        y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


if __name__ == '__main__':
    FILE_NAME_LIST = ['_train.txt.prepro', '_valid.txt.prepro', '_test.txt.prepro']

    _, _, wt, ct, _ = load_data('data', 65, flist = FILE_NAME_LIST)
    print(wt.keys())

    count = 0
    for x, y in DataReader(wt['valid'], ct['valid'], 20, 35).iter():
        count += 1
        print(x, y)
        if count > 0:
            break
