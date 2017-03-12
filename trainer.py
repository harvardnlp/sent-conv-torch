"""
Convolutional net for sentence classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import numpy as np

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('infile', help="Input file", 
                    type=argparse.FileType('r'))
parser.add_argument('-o', '--outfile', help="Output file",
                    default=sys.stdout, type=argparse.FileType('w'))

args = parser.parse_args(arguments)

parser.add_argument('data', default='', help='Training data and word2vec data')
parser.add_argument('--model_type', default='nonstatic',
                    help='Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

def cmd_option(name, default, help_str):
    parser.add_argument("-" + name, default=default, help=help_str)

# Training hyperparameters
cmd_option('-num_epochs', 25, 'Number of training epochs')
cmd_option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
cmd_option('-L2s', 3, 'L2 normalize weights')
cmd_option('-batch_size', 50, 'Batch size for training')

# Model hyperparameters
cmd_option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
cmd_option('-kernels', '[3,4,5]', 'Kernel sizes of convolutions, table format.')
cmd_option('-skip_kernel', 0, 'Use skip kernel')
cmd_option('-dropout_p', 0.5, 'p for dropout')
cmd_option('-highway_mlp', 0, 'Number of highway MLP layers')
cmd_option('-highway_conv_layers', 0, 'Number of highway MLP layers')


# cmd_option('-cudnn', 0, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
# cmd_option('-seed', 3435, 'random seed, set -1 for actual random')
# cmd_option('-folds', 10, 'number of folds to use. If test set provided, folds=1. max 10')
# cmd_option('-debug', 0, 'print debugging info including timing, confusions')
# cmd_option('-gpuid', 0, 'GPU device id to use.')
# cmd_option('-savefile', '', 'Name of output file, which will hold the trained model, model parameters, and training scores. Default filename is TIMESTAMP_results')
# cmd_option('-zero_indexing', 0, 'If data is zero indexed')
# cmd_option('-dump_feature_maps_file', '', 'Set file to dump feature maps of convolution')
# cmd:text()

# -- Preset by preprocessed data
# cmd_option('-has_test', 1, 'If data has test, we use it. Otherwise, we use CV on folds')
# cmd_option('-has_dev', 1, 'If data has dev, we use it, otherwise we split from train')
# cmd_option('-num_classes', 2, 'Number of output classes')
# cmd_option('-max_sent', 59, 'maximum sentence length')
# cmd_option('-vec_size', 300, 'word2vec vector size')
# cmd_option('-vocab_size', 18766, 'Vocab size')
# cmd:text()

# -- Training own dataset
# cmd_option('-train_only', 0, 'Set to 1 to only train on data. Default is cross-validation')
# cmd_option('-test_only', 0, 'Set to 1 to only do testing. Must have a -warm_start_model')
# cmd_option('-preds_file', '', 'On test data, write predictions to an output file. Set test_only to 1 to use')
# cmd_option('-warm_start_model', '', 'Path to .t7 file with pre-trained model. Should contain a table with key \'model\'')
# cmd:text()

def train_epoch(net, inputs, targets):
    optimizer = optim.SGD(net.parameters(), lr = 0.01)

    for i in range(0, inputs.size()[0], 50):
        input = inputs[i:i+50]
        target = targets[i:i+50]
        optimizer.zero_grad() 
        output = net(input)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

class ConvNet(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.word_lut = nn.LookupTable(opt.vocab_size, opt.vec_size)
        self.conv = [nn.TemporalConvolution(opt.vec_size, opt.num_feat_maps, opt.kernels[i])
                     for k in opt.kernels] 
        self.linear = nn.Linear(opt.num_feat_maps * len(opt.kernels), opt.num_classes)
        self.dropout = nn.Dropout(opt.dropout_p)
        
    def forward(self, x):
        x = self.word_lut(x)
        x = torch.cat([F.relu(self.conv[k](x)).data.max(2)
                       for k in opt.kernels], 2)
        x = self.linear(self.dropout(x))
        return F.log_softmax(x)
    
def main(arguments):
    opt = parser.parse_args(arguments)
    
    with h5py.File(opt.data) as f:
        train = np.array(f["train"])
        train_label = np.array(f["train_label"])

    opt.data()
    net = ConvNet(opt)
    train_epoch(train, train_label, net)
    


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
