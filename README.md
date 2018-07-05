# HMM-PoS-tagger

This project aims to create a Part of Speech (PoS) classifier, based on Hidden Markov Model (HMM), for Italian Language.
An italian version of the Universal Dependency Treebank (https://github.com/UniversalDependencies/UD_Italian-ISDT) is utilized to get train, validation and test data sets.


## About HMM and PoS Tagging

Hidden Markov Model is a classical formulation for a well known problem: given an input sequence of element (the output of a markov process) we would like
to reconstruct the hidden states.
In our case we would like to tag, i.e. assigning a specific label, to each word in a corpus from a pre-defined set of labels.
Main idea is that the probability of a word w_i to have a tag t_i (with i = 1....n) is described as:

```
P(t_i|w_i)= P(w_i|t_i)P(t_i|t_i-1)
```
where P(w_i|t_i) is the probability of seeing the w_i given the tag t_i; P(t_i|t_i-1) is the probability of seeing a tag t at time i given the tag t_i-1 at the previous time. Of course,
P(x|y) is the conditional probability of an event x given the event y.

Given a word w, the task is to find the tag which maximizes the above probability;
the problem is addressed in two steps:
- Calculate all the possible probabilities;
- Maximize the sequence of possible tags with a dynamic programming technique through the Viterbi Algorithm.

The calculation of  P(w_i|t_i) and P(t_i|t_i-1) is simply based on the count of the occurrences in the corpus. A well known issue is the
handling of unknown words. To address this issue a simple smoothing assumption was made:

```
P(unk|t_i) = 1/#(PoS_TAGs)
```

## Code

We implemented this two steps solution inside the HiddenMarkovModel class; the two main methods are *train_hmm* and *tag*.
As already underlined the *train_hmm* method simply counts the occurrences inside the corpus of various types: couple of subsequent tags,
suffixes-tags, words-tags, and so on.
The *tag* method takes as input a splitted sentence with T words, and assigns to each word a label,
taken from the Universal Dependency Treebank PoS tag list. Note that, in the *tag* method we do not perform any lemmatization on the words of the
sentence. This is because we only use the already available data sets, which contain already lemmatized sentences.

## Results
We achieved an accuracy of 0.95286 on the test set.
