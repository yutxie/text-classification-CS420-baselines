import os
import codecs
import logging as log

import torch

from nltk.tokenize import word_tokenize


def sent_tokenize(sent):
    SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"
    if isinstance(sent, str):
        return [SOS_TOK] + word_tokenize(sent) + [EOS_TOK]
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
        return [SOS_TOK] + sent + [EOS_TOK]


# def _build_vocab(args, tasks):
#     ''' Build vocabulary from scratch, reading data from tasks. '''
#     log.info("\tBuilding vocab from scratch")
#     max_v_sizes = {
#         'word': args.max_word_v_size,
#         'char': args.max_char_v_size,
#     }

#     # count words
#     word2freq, char2freq = defaultdict(int), defaultdict(int)
#     for task in tasks:
#         log.info("\tCounting words for task: '%s'", task.name)
#         for sentence in task.train_data_text[0] + task.test_data_text[0]:
#             for word in sentence:
#                 word2freq[word] += 1
#                 for char in list(word):
#                     char2freq[char] += 1
#             return
#     log.info("\tFinished counting words")

#     # build vocab
#     vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes)
#     for special in SPECIALS:
#         vocab.add_token_to_namespace(special, 'tokens')

#     words_by_freq = [(word, freq) for word, freq in word2freq.items()]
#     words_by_freq.sort(key=lambda x: x[1], reverse=True)
#     for word, _ in words_by_freq[:max_v_sizes['word']]:
#         vocab.add_token_to_namespace(word, 'tokens')

#     chars_by_freq = [(char, freq) for char, freq in char2freq.items()]
#     chars_by_freq.sort(key=lambda x: x[1], reverse=True)
#     for char, _ in chars_by_freq[:max_v_sizes['char']]:
#         vocab.add_token_to_namespace(char, 'chars')

#     # save vocab
#     vocab_path = os.path.join(args.pre_dir + 'vocab')
#     vocab.save_to_files(vocab_path)
#     log.info("\tSaved vocab to %s", vocab_path)
#     return vocab