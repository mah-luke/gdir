"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""

import os
import glob
import logging
from bs4 import BeautifulSoup
import createindex

LOGGER = logging.getLogger('createindex')


def parse_topics(data_path):
    LOGGER.info('Parsing topics')
    file_path = glob.iglob(os.path.join(data_path, 'topics.xml'))
    parse_dict = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')

        for topic in soup.find_all('topic', recursive=False):
            parse_dict[topic['id']] = createindex.text2tokens(topic.text)

    return parse_dict


def tf_idf(data_path):
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

    """
    LOGGER.info('Ranking results with TF-IDF')
    parse_dict = parse_topics(data_path)
    dataset = createindex.load(data_path)
    DF = {}

    # dataset = {
    #               t1: [(1, 1), (572, 4)]
    #               t2: ...
    #           }

    for token in dataset:
        DF[token] = len(dataset[token])

