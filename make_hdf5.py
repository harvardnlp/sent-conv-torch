import numpy as np
import h5py

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def load_data(pos_fname, neg_fname):
  max_sent_len = 0
  word_to_idx = {}
  idx = 2

  pos_file = open(pos_fname, "r")
  for line in pos_file:
      words = line.strip().split(' ')
      max_sent_len = max(max_sent_len, len(words))
      for word in words:
          if not word in word_to_idx:
              word_to_idx[word] = idx
              idx += 1

  neg_file = open(neg_fname, "r")
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

  for line in neg_file:
      words = line.strip().split(' ')
      sent = [word_to_idx[word] for word in words]
      if len(sent) < max_sent_len:
          sent.extend([1] * (max_sent_len - len(sent)))

      neg_data.append(sent)

  labels = [1] * len(pos_data) + [2] * len(neg_data)

  pos_file.close()
  neg_file.close()

  return np.array(pos_data + neg_data, dtype=np.int32), np.array(labels, np.int32), word_to_idx

if __name__ == '__main__':
  data, labels, word_to_idx = load_data("rt-polarity.pos", "rt-polarity.neg")
  w2v = load_bin_vec("/n/rush_lab/data/GoogleNews-vectors-negative300.bin", word_to_idx)
  V = len(word_to_idx) + 1
  print 'Vocab size:', V

  # Not all words in word_to_idx are in w2v.
  embed = np.random.rand(V, len(w2v.values()[0]))
  # Word embeddings initialized to random Unif(-0.25, 0.25)
  embed = (embed - 0.5) / 2
  for word, vec in w2v.items():
    embed[word_to_idx[word]] = vec

  N = data.shape[0]
  print 'data size:', N
  perm = np.random.permutation(N)
  data = data[perm]
  labels = labels[perm]

  i1 = 8*N/10
  i2 = 9*N/10
  with h5py.File("data.hdf5", "w") as f:
    f["train"] = data[:i1, ]
    f["train_label"] = labels[:i1, ]
    f["dev"] = data[i1:i2, ]
    f["dev_label"] = labels[i1:i2, ]
    f["test"] = data[i2: , ]
    f["test_label"] = labels[i2: , ]
    f["w2v"] = np.array(embed)
