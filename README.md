# Sentence Convolution Code in Torch

This code implements Yoon's sentence convolution code in torch with GPUs. It replicates his results on existing datasets, and allows training of models on arbitrary other text datasets.

## Quickstart

To make data in hdf5 format, run the following (with word2vec .bin path and choice of dataset):

    python make_hdf5.py /path/to/word2vec.bin MR

To run training with GPUs:

    th main.lua -data MR.hdf5 -cudnn 1

Results are timestamped and saved to the `results/` directory.

## Dependencies

The training pipeline requires Python hdf5 (the h5py module) and the following lua packages:
  * hdf5
  * cudnn

Training on word2vec architecture models requires downloading [word2vec](https://code.google.com/p/word2vec/) and unzipping.

## Creating datasets

We process the following datasets: `MR, SST1, SST2, Subj, TREC, CR, MPQA`.
All raw training data is located in the `data/` directory. The `SST1, SST2` data have both test and dev sets, and TREC has a test set.

The data takes word2vec embeddings (from `/n/rush_lab/data/GoogleNews-vectors-negative300.bin`), processes the vocabulary, and outputs a data matrix of vocabulary indices for each sentence.

Each dataset is packaged into a `.hdf5` file and includes the word2vec embeddings.

To create the hdf5 file, run the following with DATASET as one of the described datasets:

    python make_hdf5.py /path/to/word2vec.bin DATASET

### Training on custom datasets

We allow training on arbitrary text datasets. They should be formatted in the same way as SST-1, with one sentence per line, and the first word the class label (0-indexed). Our code handles most parsing of punctuation, possessives, capitalization, etc.

Example:

    1 no movement , no yuks , not much of anything .

Then run:

    python make_hdf5.py /path/to/word2vec.bin custom /path/to/train/data

## Running torch

Training is done with 10-fold cross-validation and 25 epochs. If the data set comes with a test set, we don't do cross validation (but split training data 90/10 for the dev set). If the data comes with the dev set, we don't do additional preprocessing.

There are four main model architectures we implemented, as described in Yoon's paper: `rand, static, nonstatic, multichannel`.
  * `rand` initializes the word embeddings randomly and learns them.
  * `static` initializes the word embeddings to word2vec and keeps the weight static.
  * `nonstatic` also initializes to word2vec, but allows them to be learned.
  * `multichannel` has two word2vec embedding layers, one static and one nonstatic. The two layers outputs are summed.

It is highly recommended that GPUs are used during training if possible (see Results section for timing benchmarks).

### Model augmentations

A few modifications were made to the model architecture as experiments.

  * we include an option to include highway layers at the final MLP step (which increases depth of the model),
  * we also include highway layers at the convolutional step (which performs multiple convolutions on the resulting feature maps) as an option,
  * we experimented with skip kernels of size 5 (added in parallel with the other kernel sizes)

Results from these experiments are described below in the Results section.

### Parameters

The following parameters are allowed by the torch code.
  * `cudnn`: Use GPUs if set to 1, otherwise set to 0
  * `num_epochs`: Number of training epochs.
  * `model_type`: Model architecture, as described above. Options: rand, static, nonstatic, multichannel
  * `data`: Training dataset to use, including word2vec data. This should be a `.hdf5` file made with `make_hdf5.py`.
  * `seed`: Random seed, set to -1 for actual randomness
  * `folds`: Number of folds for cross-validation.
  * `has_test`: Set 1 if data has test set
  * `has_dev`: Set 1 if data has dev set
  * `zero_indexing`: Set 1 if data is zero indexed
  * `debug`: Print debugging info including timing and confusions

Training parameters:
  * `optim_method`: Gradient descent method. Options: adadelta, adam
  * `L2s`: Set L2 norm of final linear layer weights to this.
  * `batch_size`: Batch size for training.

Model parameters:
  * `num_feat_maps`: Number of convolution feature maps.
  * `kernel1`, `kernel2`, `kernel3`: Kernel size of different convolutions.
  * `dropout_p`: Dropout probability.
  * `num_classes`: Number of prediction classes.
  * `highway_mlp`: Number of highway MLP layers (0 for none)
  * `highway_conv_layers`: Number of highway convolutional layers (0 for none)
  * `skip_kernel`: Set 1 to use skip kernels

### Output

When training is complete, the code outputs the following table into a file `TIMESTAMP_results.t7`:
  * `dev_scores` with dev scores,
  * `test scores` with test scores,
  * `opts` with model parameters,
  * `model` with best model (as determined by cross-validation)

## Results

The following results were collected with the same training setup as in Yoon's paper (same parameters, 10-fold cross validation if data has no test set, 25 epochs).

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

Most of this code is based on Yoon's paper and Theano [code](https://github.com/yoonkim/CNN_sentence/). 

    Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1746–1751, Doha, Qatar. Association for Computational Linguistics.

    Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training very deep networks. In Advances in Neural Information Processing Systems (pp. 2368-2376).
