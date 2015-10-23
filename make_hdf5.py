import numpy as np
import h5py
import re

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

def line_to_words(line):
  clean_line = clean_str(line.strip())
  words = clean_line.split(' ')
  return words

def load_data(pos_fname, neg_fname):
  max_sent_len = 0
  word_to_idx = {}
  idx = 2

  pos_file = open(pos_fname, "r")
  for line in pos_file:
      words = line_to_words(line)
      max_sent_len = max(max_sent_len, len(words))
      for word in words:
          if not word in word_to_idx:
              word_to_idx[word] = idx
              idx += 1

  neg_file = open(neg_fname, "r")
  for line in neg_file:
      words = line_to_words(line)
      max_sent_len = max(max_sent_len, len(words))
      for word in words:
          if not word in word_to_idx:
              word_to_idx[word] = idx
              idx += 1

  pos_file.seek(0)
  neg_file.seek(0)
  pos_data = []
  neg_data = []

  extra_padding = 7
  for line in pos_file:
      words = line_to_words(line)
      sent = [word_to_idx[word] for word in words]
      if len(sent) < max_sent_len + extra_padding:
          sent.extend([1] * (max_sent_len + extra_padding - len(sent)))

      pos_data.append(sent)

  for line in neg_file:
      words = line_to_words(line)
      sent = [word_to_idx[word] for word in words]
      if len(sent) < max_sent_len + extra_padding:
          sent.extend([1] * (max_sent_len + extra_padding - len(sent)))

      neg_data.append(sent)

  labels = [1] * len(pos_data) + [2] * len(neg_data)

  pos_file.close()
  neg_file.close()

  return np.array(pos_data + neg_data, dtype=np.int32), np.array(labels, np.int32), word_to_idx

def clean_str(string, TREC=False):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Every dataset is lower cased except for TREC
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
  string = re.sub(r"\'s", " \'s", string) 
  string = re.sub(r"\'ve", " \'ve", string) 
  string = re.sub(r"n\'t", " n\'t", string) 
  string = re.sub(r"\'re", " \'re", string) 
  string = re.sub(r"\'d", " \'d", string) 
  string = re.sub(r"\'ll", " \'ll", string) 
  string = re.sub(r",", " , ", string) 
  string = re.sub(r"!", " ! ", string) 
  string = re.sub(r"\(", " \( ", string) 
  string = re.sub(r"\)", " \) ", string) 
  string = re.sub(r"\?", " \? ", string) 
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip() if TREC else string.strip().lower()

if __name__ == '__main__':
  data, labels, word_to_idx = load_data("rt-polarity.pos", "rt-polarity.neg")
  # w2v = load_bin_vec("/n/rush_lab/data/GoogleNews-vectors-negative300.bin", word_to_idx)
  w2v = load_bin_vec("../CNN_sentence/GoogleNews-vectors-negative300.bin", word_to_idx)
  V = len(word_to_idx) + 1
  print 'Vocab size:', V

  # Not all words in word_to_idx are in w2v.
  embed = np.random.rand(V, len(w2v.values()[0]))
  # Word embeddings initialized to random Unif(-0.25, 0.25)
  embed = (embed - 0.5) / 2
  for word, vec in w2v.items():
    embed[word_to_idx[word] - 1] = vec

  N = data.shape[0]
  print 'data size:', data.shape
  perm = np.random.permutation(N)
  data = data[perm]
  labels = labels[perm]

  with h5py.File("data.hdf5", "w") as f:
    f["data"] = data
    f["data_label"] = labels
    f["w2v"] = np.array(embed)
