import os
import codecs
import logging as log

from collections import defaultdict

import jieba
import torch

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def sent_tokenize(sent, se=False):
    SOS_TOK, EOS_TOK = '<SOS>', '<EOS>'
    if isinstance(sent, str):
        sent = jieba.lcut(sent)
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
    if se: sent = [SOS_TOK] + sent + [EOS_TOK]
    return sent

def build_vocab(corpus, vocab_size):
    log.info("Start to build vocab")

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

def calc_tfidf_matrix(corpus, max_features, stop_words='english'):
    corpus = [' '.join(sent) for sent in corpus]
    vectorizer = CountVectorizer(
        # max_df= .999,
        # min_df = .001,
        max_features=max_features
    )
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus)
    )
    word_list = vectorizer.get_feature_names()
    log.info("Finished building a word list of size %i" % len(word_list))
    return tfidf, word_list