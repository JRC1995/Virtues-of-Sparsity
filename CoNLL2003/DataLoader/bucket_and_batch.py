
import numpy as np
import random
import re


class bucket_and_batch:

    def bucket_and_batch(self, source, labels,
                         vocab2idx, tags2idx, batch_size):


        PAD_vocab_index = vocab2idx['<PAD>']
        Negative_tag_idx = tags2idx['O']

        true_seq_lens = np.zeros((len(source)), dtype=int)
        for i in range(len(source)):
            true_seq_lens[i] = len(source[i])

        # sorted in descending order after flip
        sorted_by_len_indices = np.flip(np.argsort(true_seq_lens), 0)

        sorted_source = []
        sorted_labels = []

        for i in range(len(source)):

            sorted_source.append(
                source[sorted_by_len_indices[i]])

            sorted_labels.append(
                labels[sorted_by_len_indices[i]])

        i = 0
        batches_source = []
        batches_labels = []
        batches_true_seq_lens = []

        while i < len(sorted_source):

            token_length = len(sorted_source[i])
            batch_size = batch_size  # //token_length

            if i+batch_size > len(sorted_source):
                break

            batch_source = []
            batch_labels = []
            batch_true_seq_lens = []

            max_len = len(sorted_source[i])

            for j in range(i, i + batch_size):

                sample_source = sorted_source[j]
                sample_labels = sorted_labels[j]

                init_len = len(sample_source)

                while len(sample_source) < max_len:
                    sample_source.append(PAD_vocab_index)
                    sample_labels.append(Negative_tag_idx)

                batch_source.append(sample_source)
                batch_labels.append(sample_labels)
                batch_true_seq_lens.append(init_len)

            batch_source = np.asarray(batch_source, dtype=np.int32)
            batch_labels = np.asarray(batch_labels, dtype=np.int32)
            batch_true_seq_lens = np.asarray(batch_true_seq_lens, dtype=np.int32)

            batches_source.append(batch_source)
            batches_labels.append(batch_labels)
            batches_true_seq_lens.append(batch_true_seq_lens)

            i += batch_size

        return batches_source, batches_labels, batches_true_seq_lens
