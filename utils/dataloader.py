from __future__ import print_function
import torch
import pandas as pd
import numpy as np
import util as ut

class TextClassDataLoader(object):

    def __init__(self, path_file, word_to_index, batch_size=32):
        """

        Args:
            path_file:
            word_to_index:
            batch_size:
        """

        self.batch_size = batch_size
        self.word_to_index = word_to_index

        # read file
        df = pd.read_csv(path_file, delimiter=',')
        df['H_G_D'] = df['H_G_D'].apply(ut._tokenize)
        df['H_G_D'] = df['H_G_D'].apply(self.generate_indexifyer())
        self.samples = df.values.tolist()

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = int(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        self._shuffle_indices()

        self.report()

    def _shuffle_indices(self):
        self.indices = np.arange(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[2]))
        return length

    def generate_indexifyer(self):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index['__UNK__'])
            return indices

        return indexify

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        id = [item[0] for item in batch]
        h_g_d = [item[2] for item in batch]
        result = list(zip(id, h_g_d))

        # string, label = tuple(zip(*batch))
        label, string = zip(*result)

        seq_lengths = torch.LongTensor(list(map(len, string)))

        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        label = torch.LongTensor(label)

        return seq_tensor, label, seq_lengths

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
