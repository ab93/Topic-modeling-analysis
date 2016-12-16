import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
import matplotlib.pyplot as plt

def convert_gamma_to_prob(gamma_file, output_path=None):
    df = pd.read_csv(gamma_file, delim_whitespace=True, header=None)
    df = df.apply(lambda x: x/np.sum(x), axis=1, raw=True)
    probs = df.values
    return np.argsort(-probs, axis=1), -np.sort(-probs, axis=1)

def find_doc_topics_single(gamma_file, paper_id_file, output_file):
    output_dir = '/'.join(output_file.split('/')[:-1])
    gamma_df = pd.read_csv(gamma_file, delim_whitespace=True, header=None)
    num_docs = len(gamma_df)
    paper_ids = np.loadtxt(paper_id_file, dtype=int)
    doc_topics = np.argmax(gamma_df.values, axis=1)
    topic_count = Counter(doc_topics)

    topic_doc_map = defaultdict(list)
    for i in range(num_docs):
        topic_doc_map[doc_topics[i]].append(paper_ids[i])
    
    with open(output_file, 'w') as outfile:
        outfile.write('TOPIC ID' + '\t' + 'PAPER_ID' + '\n')
        for topic in topic_doc_map: 
            print topic
            for paper in topic_doc_map[topic]:
                outfile.write(str(topic) + '\t' + str(paper) + '\n')

    plt.xlabel('Topic ID')
    plt.ylabel('Number of documents')
    plt.bar(range(len(topic_count)), topic_count.values())
    plt.xticks(range(0,len(topic_count),8))
    plt.savefig(output_dir + '/distribution.png')


def find_doc_topics(gamma_file, paper_id_file, output_file, n=3):
    n = int(n)
    output_dir = '/'.join(output_file.split('/')[:-1])
    doc_topics, probs = convert_gamma_to_prob(gamma_file)
    doc_topics, probs = doc_topics[:,:n], probs[:,:n]

    num_docs = doc_topics.shape[0]
    paper_ids = np.loadtxt(paper_id_file, dtype=int)

    topic_doc_map = defaultdict(list)

    count = 0
    for i in xrange(num_docs):
        for j in range(n):
            if probs[i,j] < 0.005:
                count += 1
                continue
            topic_doc_map[doc_topics[i,j]].append((paper_ids[i], probs[i,j]))

    with open(output_file, 'w') as outfile:
        outfile.write('TOPIC ID' + '\t' + 'PAPER_ID' + '\t' + 'PROBABILITY' + '\n')
        for topic in topic_doc_map: 
            print topic
            for paper, prob in topic_doc_map[topic]:
                outfile.write(str(topic) + '\t' + str(paper) + '\t' + "{0:.5f}".format(prob) + '\n')


if (__name__ == '__main__'):
    if len(sys.argv) == 5:
        gamma_file = sys.argv[1]
        paper_id_file = sys.argv[2]
        output_file = sys.argv[3]
        n = sys.argv[4]
        find_doc_topics(gamma_file, paper_id_file, output_file, n)
    
    else:
        print 'usage: python docs.py <gamma-file> <paper-id-file> <output_file> <top_n>'
        sys.exit(1)