
# usage: python topics.py <beta file> <vocab file> <num words>
#
# <beta file> is output from the lda-c code
# <vocab file> is a list of words, one per line
# <num words> is the number of words to print from each topic

import sys

def print_topics(beta_file, vocab_file, output_file, nwords = 30):

    # get the vocabulary

    vocab = file(vocab_file, 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = map(lambda x: x.strip(), vocab)

    # for each line in the beta file

    indices = range(len(vocab))
    with open(output_file, 'w') as outfile:
        outfile.write('TOPIC ID' + '\t' + 'WORD' + '\n')
        topic_no = 0
        for topic in file(beta_file, 'r'):
            #print 'topic %03d' % topic_no
            print topic_no
            topic = map(float, topic.split())
            indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
            for i in range(nwords):
                #print '   %s' % vocab[indices[i]]
                outfile.write(str(topic_no) + '\t' + vocab[indices[i]] + '\n')
            topic_no = topic_no + 1
            #print '\n'

if (__name__ == '__main__'):

    if (len(sys.argv) != 5):
       print 'usage: python topics.py <beta-file> <vocab-file> <output_file> <num words>\n'
       sys.exit(1)

    beta_file = sys.argv[1]
    vocab_file = sys.argv[2]
    output_file = sys.argv[3]
    nwords = int(sys.argv[4])
    print_topics(beta_file, vocab_file, output_file, nwords)