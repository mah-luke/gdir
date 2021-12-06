"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""
import numpy
import os
import glob
import logging
from bs4 import BeautifulSoup
import createindex
import numpy as np
from multiprocessing import Pool
import collections
from collections import defaultdict


LOGGER = logging.getLogger('evaluation_mode')


def parse_topics(data_path):
    LOGGER.info('Parsing topics')
    file_path = os.path.join(data_path, 'topics.xml')
    parse_dict = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

        with Pool(6) as pool:
            for topic in soup.find('inex-topic-file').find_all('topic', recursive=False):
                parse_dict[topic['id']] = createindex.text2tokens(topic.find('title').text, pool)

    return parse_dict


def document_number(path: str):
    titles = []

    for file_path in glob.iglob(os.path.join(path, "wikipedia articles", "1.xml")):
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            for title in soup.find_all('title', recursive=False):
                titles.append(title)

    return len(titles)


def tf_idf(data_path: str):
    """

    :param data_path: path to apply algorithm to
    :return: outputs the top-100 documents in the following format
    {topic-id}  Q0  {document-id}  {rank}  {score}  {run-name}

    topic-id is the id per query from the topics file

    document-id is an identifier for the Wikipedia article

    Q0 is just a legacy hardcoded string

    rank is an integer indicating the rank of the doc in the sorted list. Starts at 1

    score the similarity score

    run-name a name you give to your experiment

    FORMULA for TF-IDF:
    tf(t,d) = count of t in d / number of words in d

    df(t) = occurrence of t in N documents

    idf(t) = log(N/(df + 1))

    tf-idf(t, d) = tf(t, d) * idf(t)

    dataset = {
                t1: [(1, 1), (572, 4)]
                t2: ...
               }

    """
    LOGGER.info('Ranking results with TF-IDF')
    dataset = createindex.load(os.path.join('out', 'inverted_index.npy'))
    df = {}
    topics = parse_topics('GIR2021 dataset')
    word_count = {}

    for token, arr in dataset.items():
        for tup in arr:
            if tup['docid'] not in word_count:
                word_count[tup['docid']] = tup['tf']
            else:
                word_count[tup['docid']] += tup['tf']

    TF_IDF = {}
    N = len(word_count)

    # Calculate tf-idf for query
    tf_idf_query = {}
    for topic_id, tokens in topics.items():
        scores = []
        scores_normalized = []
        for token in tokens:
            if token in dataset:
                idf = np.log(N / (len(dataset[token]) + 1))
                tf_query = collections.Counter(tokens)[token]/len(tokens)
                scores.append(np.log(1+tf_query) * idf)

        for score in scores:
            scores_normalized.append(score / np.linalg.norm(scores, ord=1))

        tf_idf_query[topic_id] = np.array(scores_normalized, dtype=numpy.float32)


    # Calculate tf-idf for documents
    for topic_id, tokens in topics.items():
        scores = []
        doc_dict = {}
        for i in range(len(tokens)):
            token = tokens[i]
            if token in dataset:
                idf = np.log(1+(N / (len(dataset[token]))))
                for tup in dataset[token]:
                    if tup['docid'] not in doc_dict:
                        doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                    tf_document = tup['tf'] / word_count[tup['docid']]
                    #scores.append((tup['docid'], token, (np.log(1+tf_document)*idf)))
                    doc_dict[tup['docid']][i] = np.log(1+tf_document)*idf

        # topicid: {docid: [0.009, 0.122]}

        # tfidf query topicid: [0.08, 0.9]
        #
        # docid: [0.08, 0.9] * [0.009, 0.122] = [x, y] x+y = score
        for doc_id, arr in doc_dict.items():
            doc_dict[doc_id] = arr/np.linalg.norm(arr, ord=1)

        TF_IDF[topic_id] = doc_dict

    return TF_IDF


if __name__ == "__main__":
    logging.basicConfig(level=10)
    tifu = tf_idf('ha')
    print(tifu)