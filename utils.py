import random
import numpy as np
import torch


def data_set(data_url, vocab_size):
    """process data input."""
    data_list = []
    word_count = []
    with open(data_url) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            id_freqs = line.split()
            id_freqs = id_freqs[0:-1]
            doc = {}
            count = 0
            for id_freq in id_freqs:
                items = id_freq.split(':')
                doc[int(items[0]) - 1] = int(items[1])
                count += int(items[1])
            if count > 0:
                data_list.append(doc)
                word_count.append(count)
    data_mat = np.zeros((len(data_list), vocab_size), dtype=np.int16)
    for doc_idx, doc in enumerate(data_list):
        for word_idx, count in doc.items():
            data_mat[doc_idx, word_idx] += count
    return data_list, data_mat, word_count


def create_batches(data_size, batch_size, shuffle=True):
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    for i in range(int(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
    # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        batches.append(ids[-rest:] + [-1] * (batch_size - rest))  # -1 as padding
    return batches


def fetch_data(data, count, idx_batch, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():
                data_batch[i, word_id] = freq
            count_batch.append(count[doc_id])
            mask[i] = 1.0
        else:
            count_batch.append(0)
    return data_batch, count_batch, mask


def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
    return ret_list
