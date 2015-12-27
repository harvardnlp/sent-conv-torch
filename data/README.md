# Datasets

**Data** | Classes | Average sentence length | Dataset size | Vocab size | Number of words present in word2vec | Test size
--- | --- | --- | --- | --- | --- | ---
MR | 2 | 20 | 10662 | 18765 | 16448 | CV
SST1 | 5 | 18 | 11855 | 17836 | 16262 | 2210
SST2 | 2 | 19 | 9613 | 16185 | 14838 | 1821
Subj | 2 | 23 | 10000 | 21323 | 17913 | CV
TREC | 6 | 10 | 5952 | 9592 | 9125 | 500
CR | 2 | 19 | 3775 | 5340 | 5046 | CV
MPQA | 2 | 3 | 10606 | 6246 | 6083 | CV

The following datasets are included in this directory:
  * **MR**: Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews (Pang and Lee, 2005). [Link](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
  * **SST-1**: Stanford Sentiment Treebank - an extension of MR but with train/dev/test splits provided and fine-grained labels (very positive, positive, neutral, negative, very negative), re-labeled by Socher et al. (2013). [Link](http://nlp.stanford.edu/sentiment/)

    Note that data is actually provided at the phrase-level and hence we train the model on both phrases and sentences but only score on sentences at test time, as in Socher et al. (2013), Kalchbrenner et al. (2014), and Le and Mikolov (2014). Thus the training set is an order of magnitude larger than listed in the above table.
  * **SST-2** Same as SST-1 but with neutral reviews removed and binary labels.
  * **Subj**: Subjectivity dataset where the task is to classify a sentence as being subjective or objective (Pang and Lee, 2004). [Link](http://cogcomp.cs.illinois.edu/Data/QA/QC/)
  * **TREC**: TREC question dataset - task involves classifying a question into 6 question types (whether the question is about person, location, numeric information, etc.) (Li and Roth, 2002). [Link](http://www.cs.uic.edu/⇠liub/FBS/sentiment-analysis.html)
  * **CR**: Customer reviews of various products (cameras, MP3s etc.). Task is to predict positive/negative reviews (Hu and Liu, 2004). [Link](http://www.cs.pitt.edu/mpqa/)

## Data files

--- | ---
MR | rt-polarity.all
SST-1 | stsa.fine.\*
SST-2 | stsa.binary.\*
Subj | subj.all
TREC | TREC.\*
CR | custrev.all
MPQA | mpqa.all
