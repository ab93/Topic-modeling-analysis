import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
import matplotlib.pyplot as plt

def doc_topics(gamma_file, paper_id_file, output_file):
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
    plt.xticks(range(0,len(topic_count),5))
    plt.savefig(output_dir + '/distribution.png')

if (__name__ == '__main__'):
    if (len(sys.argv) != 4):
        print 'usage: python topics.py <gamma-file> <paper-id-file> <output_file>\n'
        sys.exit(1)
            
    gamma_file = sys.argv[1]
    paper_id_file = sys.argv[2]
    output_file = sys.argv[3]
    doc_topics(gamma_file, paper_id_file, output_file)