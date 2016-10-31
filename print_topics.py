import sys
import numpy as np
import pandas as pd
from collections import defaultdict

def print_topics(word_count_file, vocab_file, topics_filename, top_n=15):
    word_counts = np.loadtxt(word_count_file)
    vocab = pd.read_table(vocab_file)
    num_vocab = len(vocab)
    num_topics = word_counts.shape[0]
    #prob = np.zeros_like(word_counts)
    topic_words_map = defaultdict(list)


    for k in range(num_topics):
        word_prob = word_counts[k,:]/np.sum(word_counts[k,:])
        top_word_idx = word_prob.argsort()[::-1][:int(top_n)]
        word_prob = word_prob[top_word_idx]
        top_words = [x for sublist in vocab.values[top_word_idx] for x in sublist]
        topic_words_map[k] = top_words

    return topic_words_map

print_topics(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
