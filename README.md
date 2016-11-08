# Sentence Convolution Code in Torch

This code implements Kim (2014) sentence convolution code in Torch with GPUs. It replicates the results on existing datasets, and allows training of models on arbitrary other text datasets.

## Quickstart

To make data in hdf5 format, run the following (with word2vec .bin path and choice of dataset):

    python preprocess.py MR /path/to/word2vec.bin

To run training with GPUs:

    th main.lua -data MR.hdf5 -cudnn 1 -gpuid 1

Results are timestamped and saved to the `results/` directory.

## Dependencies

The training pipeline requires Python hdf5 (the h5py module) and the following lua packages:
  * hdf5
  * cudnn

Training on word2vec architecture models requires downloading [word2vec](https://code.google.com/p/word2vec/) and unzipping. Simply run the script

    ./get_word2vec.sh

## Creating datasets

We provide the following datasets: `MR, SST1, SST2, SUBJ, TREC, CR, MPQA`.
All raw training data is located in the `data/` directory. The `SST1, SST2` data have both test and dev sets, and TREC has a test set.

The data takes word2vec embeddings, processes the vocabulary, and outputs a data matrix of vocabulary indices for each sentence.

To create the hdf5 file, run the following with DATASET as one of the described datasets:

    python preprocess.py DATASET /path/to/word2vec.bin

The script outputs:
  * the `DATASET.hdf5` file with the data matrix and word2vec embeddings
  * a `DATASET.txt` file with a word-index dictionary for the word embeddings

### Training on custom datasets

We allow training on arbitrary text datasets. They should be formatted in the same way as the sample data, with one sentence per line, and the first word the class label (0-indexed). Our code handles most parsing of punctuation, possessives, capitalization, etc.

Example line:

    1 no movement , no yuks , not much of anything .

Then run:

    python preprocess.py custom /path/to/word2vec.bin --train /path/to/train/data --test /path/to/test/data --dev /path/to/dev/data

The output file's name can be set with the flag `--custom_name` (default is named custom).

## Running torch

Training is typically done with 10-fold cross-validation and 25 epochs. If the data set comes with a test set, we don't do cross validation (but split training data 90/10 for the dev set). If the data comes with the dev set, we don't do the split for train/dev.

There are four main model architectures we implemented, as described in Kim (2014): `rand, static, nonstatic, multichannel`.
  * `rand` initializes the word embeddings randomly and learns them.
  * `static` initializes the word embeddings to word2vec and keeps the weight static.
  * `nonstatic` also initializes to word2vec, but allows them to be learned.
  * `multichannel` has two word2vec embedding layers, one static and one nonstatic. The two layers outputs are summed.

It is highly recommended that GPUs are used during training if possible (see Results section for timing benchmarks).

Separating out training and testing is easy; use the parameters `-train_only` and `-test_only`. Also, pretrained models at any stage can be loaded from a `.t7` file with `-warm_start_model` (see more parameters below).

### Output

The code outputs a checkpoint `.t7` file for every fold with name -savefile. The default name is `TIMESTAMP_results`.

The following are saved as a table:
  * `dev_scores` with dev scores,
  * `test scores` with test scores,
  * `opt` with model parameters,
  * `model` with best model (as determined by dev score),
  * `embeddings` with the updated word embeddings

### Model augmentations

A few modifications were made to the model architecture as experiments.

  * we include an option to include highway layers at the final MLP step (which increases depth of the model),
  * we also include highway layers at the convolutional step (which performs multiple convolutions on the resulting feature maps) as an option,
  * we experimented with skip kernels of size 5 (added in parallel with the other kernel sizes)

Results from these experiments are described below in the Results section.

### Parameters

The following is a list of complete parameters allowed by the torch code.
  * `model_type`: Model architecture, as described above. Options: rand, static, nonstatic, multichannel
  * `data`: Training dataset to use, including word2vec data. This should be a `.hdf5` file made with `preprocess.py`.
  * `cudnn`: Use GPUs if set to 1, otherwise set to 0
  * `seed`: Random seed, set to -1 for actual randomness
  * `folds`: Number of folds for cross-validation.
  * `debug`: Print debugging info including timing and confusions
  * `savefile`: Name of output `.t7` file, which will hold the trained model. Default is `TIMESTAMP_results`
  * `zero_indexing`: Set to 1 if data is zero indexed
  * `warm_start_model`: Load a `.t7` file with pretrained model. Should contain a table with key 'model'
  * `train_only`: Set to 1 to only train (no testing)
  * `test_only`: Given a `.t7` file with model, test on testing data
  * `dump_feature_maps_file`: Filename for dumping feature maps of convolution at test time. This will be a `.hdf5` file with fields `feature_maps` for the features at each time step and `word_idxs` for the word indexes (aligned with the last word of the filter). This currently only works for models with a single filter size. This is saved for the best model on fold 1.

Training hyperparameters:
  * `num_epochs`: Number of training epochs.
  * `optim_method`: Gradient descent method. Options: adadelta, adam
  * `L2s`: Set L2 norm of final linear layer weights to this.
  * `batch_size`: Batch size for training.

Model hyperparameters:
  * `num_feat_maps`: Number of convolution feature maps.
  * `kernels`: Kernel sizes of different convolutions.
  * `dropout_p`: Dropout probability.
  * `highway_mlp`: Number of highway MLP layers (0 for none)
  * `highway_conv_layers`: Number of highway convolutional layers (0 for none)
  * `skip_kernel`: Set 1 to use skip kernels

## Results

The following results were collected with the same training setup as in Kim (2014) (same parameters, 10-fold cross validation if data has no test set, 25 epochs).

### Scores

Dataset | `rand` | `static` | `nonstatic` | `multichannel`
--- | --- | --- | --- | ---
MR | 75.9 | 80.5 | 81.3 | 80.8
SST1 | 42.2 | 44.8 | 46.7 | 44.6
SST2 | 83.5 | 85.6 | 87.0 | 87.1
Subj | 89.2 | 93.0 | 93.4 | 93.2
TREC | 88.2 | 91.8 | 92.8 | 91.8
CR | 78.3 | 83.3 | 84.4 | 83.7
MPQA | 84.6 | 89.6 | 89.7 | 89.6

With 5 trials on SST1, we have a mean nonstatic score of 46.7 with standard deviation 1.69.

With 1 highway layer, SST1 achieves a mean score of mean 47.8, stddev 0.857, over 5 trials, and with 2 highway layers, mean 47.1, stddev 1.47, over 10 trials.

### Timing

We ran timing benchmarks on SST1, which has train/dev/test data sizes of 156817/1101/2210. We used a batch size of 50.

 | non-GPU | GPU
--- | --- | ---
per epoch | 2475 s | 54.0 s
per batch | 787 ms | 15.6 ms

From these results, we see that using GPUs achieves almost a 50x speedup on training. This allows much faster tuning of parameters and model experimentation.

## Relevant publications

This code is based on Kim (2014) and the corresponding Theano [code](https://github.com/yoonkim/CNN_sentence/). 

    Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1746–1751, Doha, Qatar. Association for Computational Linguistics.

    Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training very deep networks. In Advances in Neural Information Processing Systems (pp. 2368-2376).
