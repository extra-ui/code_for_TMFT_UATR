import torch
import pandas as pd
import numpy as np
import util as ut
from vocab import VocabBuilder
from torch.nn.utils.rnn import pad_sequence

def process_text_data(path_file, word_to_index, batch_size=32):
    """
    Process text data and return tensors for sequences, labels, and sequence lengths.

    Args:
        path_file (str): Path to the CSV file containing the data.
        word_to_index (dict): Dictionary mapping words to their respective indices.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        tuple: A tuple containing tensors for sequences, labels, and sequence lengths.
    """

    # Read file
    df = pd.read_csv(path_file, delimiter=',')

    df['combined'] = df.apply(lambda row: ' '.join(
        ['The type is', str(row['Type']), 'and', str(row['H_G_D']), 'Chanel_Depth is', str(row['Chanel_Depth']),
         'Wind is', str(row['Wind']),
         'Distance is', str(row['Distance'])]), axis=1)
    df['combined'] = df['combined'].apply(ut._tokenize)
    df['combined'] = df['combined'].apply(generate_indexifyer(word_to_index))
    samples = df.values.tolist()

    n_samples = len(samples)
    max_length = _get_max_length(samples)
    indices = np.arange(n_samples)

    seq_tensors = []
    labels = []
    seq_lengths = []
    print('max len: {}'.format(max_length))
    for i in range(0, n_samples, batch_size):
        batch = samples[i:i+batch_size]
        id = [item[0] for item in batch]
        h_g_d = [item[10] for item in batch]
        result = list(zip(id, h_g_d))

        label, string = zip(*result)
        seq_lengths_batch = torch.full((len(string),), max_length, dtype=torch.long)
        seq_tensor_batch = pad_sequence([torch.LongTensor(seq + [0] * (max_length - len(seq))) for seq in string], batch_first=True)

        seq_tensors.append(seq_tensor_batch)
        labels.append(torch.LongTensor(label))
        seq_lengths.append(seq_lengths_batch)

    return seq_tensors, labels, seq_lengths


def generate_indexifyer(word_to_index):
    """
    Generate a function to convert words to indices using a given word_to_index dictionary.

    Args:
        word_to_index (dict): Dictionary mapping words to their respective indices.

    Returns:
        function: A function to indexify a list of words.
    """
    def indexify(lst_text):
        indices = []
        for word in lst_text:
            if word in word_to_index:
                indices.append(word_to_index[word])
            else:
                indices.append(word_to_index['__UNK__'])
        return indices

    return indexify


def _get_max_length(samples):
    """
    Get the maximum length of sequences in the given samples.

    Args:
        samples (list): List of samples.

    Returns:
        int: Maximum length of sequences.
    """
    max_length = 0
    for sample in samples:
        max_length = max(max_length, len(sample[10]))
    return max_length


if __name__ == '__main__':
    pass
