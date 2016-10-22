import sys
import numpy as np
import pandas as pd

def calculate_topic_distribution(word_assignments_file):
    doc_topic_df = pd.read_csv(word_assignments_file, delim_whitespace=True)
    doc_topic_map = {}
    topic_doc_map = {}
    doc_paperid_map = {}
    

if __name__ == '__main__':
    calculate_topic_distribution(sys.argv[1])
