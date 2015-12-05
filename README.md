# Sentence Convolution Code in Torch

## Quickstart

To make data in hdf5 format, run the following (with choice of dataset):

  python make_hdf5.py MR

To run training with GPUs:

  th main.lua -data MR.hdf5 -cudnn 1

Results are saved to results/ directory.

## Creating datasets

We process the following datasets:

  MR, SST-1, SST-2, Subj, TREC, CR, MPQA

The data takes word2vec embeddings (from /n/rush_lab/data/GoogleNews-vectors-negative300.bin), processes the vocabulary, and outputs a data matrix of vocabulary indices for each sentence.

## Running torch

Testing is done with 10-fold cross-validation and 25 epochs (default). If test set is present, we don't do cross validation (but split training data 90/10 for dev set). If dev set is present, use the data as is. To use these, set parameters -has_dev 1 and -has_test 1

### Parameters

asdf

Default number of predicted classes is 2. To run on SST-1, set -num_classes 5, and for TREC set -num_classes 6.


Datasets with test only: TREC

Datasets with test and dev: SST-1, SST-2

## Results

When training is complete, the binary outputs the following into a file TIMESTAMP\_results:

TODO

