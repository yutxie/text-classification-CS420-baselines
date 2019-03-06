import os
import codecs
import logging as log

from collections import defaultdict

import torch

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def sent_tokenize(sent):
    SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"
    if isinstance(sent, str):
        return [SOS_TOK] + word_tokenize(sent) + [EOS_TOK]
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
        return [SOS_TOK] + sent + [EOS_TOK]


def build_vocab(corpus, vocab_size):
    log.info("Starting to build vocab")

    # count words
    word2freq = defaultdict(int)
    for sent in corpus:
        for word in sent:
            word2freq[word] += 1
    log.info("Finished counting words")

    # build vocab
    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)
    vocab_size = min(vocab_size, len(words_by_freq))
    vocab = [word for word, _ in words_by_freq[:vocab_size]]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    log.info("Finished building a vocab of size %i" % vocab_size)
    return vocab, word2idx

def calc_tfidf_matrix(corpus, max_features):
    tfidf = TfidfVectorizer(
        max_df=.999, min_df=.001, 
        max_features=max_features, stop_words='english')
    mat = tfidf.fit_transform(corpus)
    word_list = tfidf.get_feature_names()
    log.info("Finished building a word list of size %i" % len(word_list))
    return mat, word_list