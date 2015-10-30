import numpy as np
import h5py
import re
import sys

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

def line_to_words(line, dataset):
  trec = (dataset == 'TREC')
  clean_line = clean_str(line.strip(), trec)
  words = clean_line.split(' ')
  if dataset == 'SST1' or dataset == 'SST2':
    words = words[1:]
  elif dataset == 'TREC':
    words = words[2:]

  return words

def get_vocab(file_list, dataset=''):
  max_sent_len = 0
  word_to_idx = {}
  idx = 2

  for filename in file_list:
    f = open(filename, "r")
    for line in f:
        words = line_to_words(line, dataset)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = idx
                idx += 1

    f.close()

  return max_sent_len, word_to_idx

def load_data(pos_fname, neg_fname, dataset):
  max_sent_len, word_to_idx = get_vocab([pos_fname, neg_fname])

  pos_file = open(pos_fname, "r")
  neg_file = open(neg_fname, "r")

  pos_data = []
  neg_data = []

  extra_padding = 7
  for data, file in zip([pos_data, neg_data], [pos_file, neg_file]):
    for line in file:
        words = line_to_words(line, dataset)
        sent = [word_to_idx[word] for word in words]
        if len(sent) < max_sent_len + extra_padding:
            sent.extend([1] * (max_sent_len + extra_padding - len(sent)))

        data.append(sent)

  labels = [1] * len(pos_data) + [2] * len(neg_data)

  pos_file.close()
  neg_file.close()

  return np.array(pos_data + neg_data, dtype=np.int32), np.array(labels, np.int32), word_to_idx

def load_sst_data(dataset):
  """
  Load SST data. If SST1, we use 5 sentiment classes. If
  SST2, we discard neutral and use binary classes.
  """
  f_prefix = 'data/'
  if dataset == 'SST1':
    f_prefix = f_prefix + 'stsa.fine'
  elif dataset == 'SST2':
    f_prefix = f_prefix + 'stsa.binary'

  dev_name = f_prefix + '.dev'
  train_name = f_prefix + '.train'
  test_name = f_prefix + '.test'
  max_sent_len, word_to_idx = get_vocab([dev_name, train_name, test_name], dataset)

  f_dev = open(dev_name, 'r')
  f_train = open(train_name, 'r')
  f_test = open(test_name, 'r')

  dev = []
  dev_label = []
  train = []
  train_label = []
  test = []
  test_label = []

  extra_padding = 7
  for data, label, f in zip([dev, train, test], [dev_label, train_label, test_label], [f_dev, f_train, f_test]):
    for line in f:
        words = line_to_words(line, dataset)
        y = int(line[0]) + 1
        sent = [word_to_idx[word] for word in words]
        if len(sent) < max_sent_len + extra_padding:
            sent.extend([1] * (max_sent_len + extra_padding - len(sent)))

        data.append(sent)
        label.append(y)

  f_dev.close()
  f_train.close()
  f_test.close()

  return np.array(train, dtype=np.int32), np.array(train_label, dtype=np.int32), np.array(dev, dtype=np.int32), np.array(dev_label, dtype=np.int32), np.array(test, dtype=np.int32), np.array(test_label, dtype=np.int32), word_to_idx

def load_trec_data(dataset='TREC'):
  """
  Load TREC data
  """
  f_prefix = 'data/TREC'

  train_name = f_prefix + '.train'
  test_name = f_prefix + '.test'
  max_sent_len, word_to_idx = get_vocab([train_name, test_name], dataset)

  f_train = open(train_name, 'r')
  f_test = open(test_name, 'r')

  train = []
  train_label = []
  test = []
  test_label = []

  extra_padding = 7
  c = {'DESC': 1, 'ENTY': 2, 'ABBR': 3, 'HUM': 4, 'LOC': 5, 'NUM': 6}
  for data, label, f in zip([train, test], [train_label, test_label], [f_train, f_test]):
    for line in f:
        words = line_to_words(line, dataset)
        y = c[line.split(':')[0]]
        sent = [word_to_idx[word] for word in words]
        if len(sent) < max_sent_len + extra_padding:
            sent.extend([1] * (max_sent_len + extra_padding - len(sent)))

        data.append(sent)
        label.append(y)

  f_train.close()
  f_test.close()

  return np.array(train, dtype=np.int32), np.array(train_label, dtype=np.int32), np.array(test, dtype=np.int32), np.array(test_label, dtype=np.int32), word_to_idx

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
  string = re.sub(r"\(", " ( ", string) 
  string = re.sub(r"\)", " ) ", string) 
  string = re.sub(r"\?", " ? ", string) 
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower() if TREC else string.strip()

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print 'Must specify dataset'
    sys.exit(0)

  # Dataset name
  dataset = sys.argv[1]
  if dataset == 'MR':
    data, labels, word_to_idx = load_data("rt-polarity.pos", "rt-polarity.neg", dataset)
  elif dataset == 'Subj':
    data, labels, word_to_idx = load_data("data/subj.objective", "data/subj.subjective", dataset)
  elif dataset == 'CR':
    data, labels, word_to_idx = load_data("data/custrev.pos", "data/custrev.neg", dataset)
  elif dataset == 'MPQA':
    data, labels, word_to_idx = load_data("data/mpqa.pos", "data/mpqa.neg", dataset)
  elif dataset == 'TREC':
    train, train_label, test, test_label, word_to_idx = load_trec_data()
  elif dataset == 'SST1' or dataset == 'SST2':
    train, train_label, dev, dev_label, test, test_label, word_to_idx = load_sst_data(dataset)
  else:
    print 'Unrecognized dataset:', dataset
    sys.exit(0)

  w2v = load_bin_vec("/n/rush_lab/data/GoogleNews-vectors-negative300.bin", word_to_idx)
  # w2v = load_bin_vec("../CNN_sentence/GoogleNews-vectors-negative300.bin", word_to_idx)
  V = len(word_to_idx) + 1
  print 'Vocab size:', V

  # Not all words in word_to_idx are in w2v.
  # Word embeddings initialized to random Unif(-0.25, 0.25)
  embed = np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0])))
  for word, vec in w2v.items():
    embed[word_to_idx[word] - 1] = vec

  if dataset == 'MR' or dataset == 'Subj' or dataset == 'CR' or dataset == 'MPQA':
    N = data.shape[0]
    print 'data size:', data.shape
    perm = np.random.permutation(N)
    data = data[perm]
    labels = labels[perm]
  else:
    print 'train size:', train.shape

  filename = dataset + '.hdf5'
  with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(embed)
    if dataset == 'TREC':
      f['train'] = train
      f['train_label'] = train_label
      f['test'] = test
      f['test_label'] = test_label
    elif dataset == 'SST1' or dataset == 'SST2':
      f['train'] = train
      f['train_label'] = train_label
      f['test'] = test
      f['test_label'] = test_label
      f['dev'] = dev
      f['dev_label'] = dev_label
    else:
      f["data"] = data
      f["data_label"] = labels
