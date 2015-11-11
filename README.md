Sentence Convolution Code in Torch

To make all data in hdf5 format:

  python make_hdf5.py

To run with GPUs:

  th main.lua -data MR.hdf5 -cudnn 1

Results are saved to results/ directory.

Default number of predicted classes is 2. To run on SST-1, set -num_classes 5, and for TREC set -num_classes 6.

Testing is done with 10-fold cross-validation and 25 epochs (default). If test set is present, we don't do cross validation (but split training data 90/10 for dev set). If dev set is present, use the data as is. To use these, set parameters -has_dev 1 and -has_test 1

Datasets with test only: TREC

Datasets with test and dev: SST-1, SST-2
