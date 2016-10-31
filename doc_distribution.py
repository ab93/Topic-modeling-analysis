import sys
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

YEAR = sys.argv[1].split('/')[-2]

def map_doc_to_paper(paper_id_file):
    doc_to_paper = {}
    with open(paper_id_file,'r') as f:
        idx = 0
        for row in f:
            paper = row.strip()
            doc_to_paper[idx] = paper
            idx += 1
    print "Mapped doc ID to paper ID"
    return doc_to_paper, idx

def calculate_topic_distribution(word_assignments_file, paper_id_file):
    doc_topic_df = pd.read_csv(word_assignments_file, delim_whitespace=True)
    doc_topic_df = doc_topic_df[['d','z']]
    doc_topic_df = doc_topic_df.drop_duplicates()

    doc_topic_map = defaultdict(list)
    topic_doc_map = defaultdict(list)
    doc_paperid_map, num_docs = map_doc_to_paper(paper_id_file)

    for i in range(len(doc_topic_df)):
        doc, topic = doc_topic_df.iloc[i]
        doc_topic_map[doc_paperid_map[doc]].append(topic)
        topic_doc_map[topic].append(doc_paperid_map[doc])

    topic_prob = np.zeros(len(topic_doc_map))
    for topic in topic_doc_map:
        topic_prob[topic] = len(topic_doc_map[topic])

    topic_prob = np.divide(topic_prob,num_docs)
    plt.bar(np.arange(len(topic_prob)),topic_prob, align='center')
    plt.title('Topic distributions among docs in ' + YEAR)
    plt.xlabel('Topic')
    plt.ylabel('Probability')
    plt.xticks(np.arange(0,len(topic_prob),2))
    plt.xlim(0,len(topic_prob))
    plt.savefig('../results/' + YEAR + '/plot.png')

    print "Generated plots"

    return topic_doc_map



def write_results(topic_doc_map, top_n=20):
    topic_words_map = print_topics('../hdp_results/'+ YEAR +'/mode-topics.dat',
                                    '../lda_data/vocab' + YEAR + '.txt',
                                    '../topics/' + YEAR +'.txt')
    topic_doc_map = OrderedDict(sorted(topic_doc_map.items()))

    with open('../results/' + YEAR + '/papers_per_topic.txt','w+') as outfile:
        outfile.write('TOPIC ID' + '\t' + 'PAPER ID' + '\n')
        for topic_id in topic_doc_map:
            count = 0
            for doc in topic_doc_map[topic_id]:
                if count >= top_n:
                    break
                outfile.write(str(topic_id) + '\t' + str(doc) + '\n')
            #outfile.write('\n')

    with open('../results/' + YEAR + '/words_per_topic.txt','w+') as outfile:
        outfile.write('TOPIC ID' + '\t' + 'WORD' + '\n')
        for topic_id in topic_doc_map:
            for word in topic_words_map[topic_id]:
                outfile.write(str(topic_id) + '\t' + word + '\n')
            #outfile.write('\n')


def print_topics(word_count_file, vocab_file, topics_filename, top_n=15):
    word_counts = np.loadtxt(word_count_file)
    vocab = pd.read_table(vocab_file, header=None)
    num_vocab = len(vocab)
    num_topics = word_counts.shape[0]
    #prob = np.zeros_like(word_counts)
    topic_words_map = defaultdict(list)
    print word_counts.shape
    print num_vocab
    #raw_input()
    vocab.loc[num_vocab] = 'none'


    for k in range(num_topics):
        print k
        #raw_input()
        word_prob = word_counts[k,:]/np.sum(word_counts[k,:])
        top_word_idx = word_prob.argsort()[::-1][:int(top_n)]
        word_prob = word_prob[top_word_idx]
        top_words = [x for sublist in vocab.values[top_word_idx] for x in sublist]
        topic_words_map[k] = top_words

    print "Calculated topic words map"
    return topic_words_map


if __name__ == '__main__':
    word_assignments_file = sys.argv[1]
    paper_id_file = sys.argv[2]
    topic_doc_map = calculate_topic_distribution(word_assignments_file, paper_id_file)
    write_results(topic_doc_map)
