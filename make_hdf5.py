import numpy as np
import h5py

max_sent_len = 0
word_to_idx = {}
idx = 2

pos_file = open("rt-polarity.pos", "r")
for line in pos_file:
    words = line.strip().split(' ')
    max_sent_len = max(max_sent_len, len(words))
    for word in words:
        if not word in word_to_idx:
            word_to_idx[word] = idx
            idx += 1

neg_file = open("rt-polarity.neg", "r")
for line in neg_file:
    words = line.strip().split(' ')
    max_sent_len = max(max_sent_len, len(words))
    for word in words:
        if not word in word_to_idx:
            word_to_idx[word] = idx
            idx += 1

pos_file.seek(0)
neg_file.seek(0)
pos_data = []
neg_data = []

for line in pos_file:
    words = line.strip().split(' ')
    sent = [word_to_idx[word] for word in words]
    if len(sent) < max_sent_len:
        sent.extend([1] * (max_sent_len - len(sent)))

    pos_data.append(sent)

for i,line in enumerate(neg_file):
    words = line.strip().split(' ')
    sent = [word_to_idx[word] for word in words]
    if len(sent) < max_sent_len:
        sent.extend([1] * (max_sent_len - len(sent)))

    neg_data.append(sent)

labels = [1] * len(pos_data) + [2] * len(neg_data)

with h5py.File("rt-polarity.hdf5", "w") as f:
    f["train"] = np.array(pos_data + neg_data, dtype=np.int32)
    f["train_label"] = np.array(labels)

print 'Vocab size:', len(word_to_idx)
# TODO(jeffreyling): Need to write word_to_idx
