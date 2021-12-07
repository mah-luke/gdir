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
import json


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


def tf_idf():
    """

    :return: outputs the top-100 documents in the following format
    {topic-id}  Q0  {document-id}  {rank}  {score}  {run-name}

    topic-id is the id per query from the topics file

    document-id is an identifier for the Wikipedia article

    Q0 is just a legacy hardcoded string

    rank is an integer indicating the rank of the doc in the sorted list. Starts at 1

    score the similarity score

    run-name a name you give to your experiment

    FORMULA for TF-IDF:
    tf(t,d) = log(1 + count of t in d / number of words in d)

    df(t) = occurrence of t in N documents

    idf(t) = log(1+(N/(df)))

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

    #Calculate tf-idf for query
    tf_idf_query = {}
    for topic_id, tokens in topics.items():
        scores = []
        scores_normalized = []
        for i in range(len(tokens)):
            token = tokens[i]
            if token in dataset:
                idf = np.log(N / len(dataset[token]))
                tf_query = collections.Counter(tokens)[token]/len(tokens)
                scores.append(np.log(1+tf_query) * idf)
            else:
                scores.append(0)

        tf_idf_query[topic_id] = np.array(scores, dtype=numpy.float32)


    # Calculate tf-idf for documents
    for topic_id, tokens in topics.items():
        scores = []
        doc_dict = {}
        for i in range(len(tokens)):
            token = tokens[i]
            if token in dataset:
                idf = np.log(N / (len(dataset[token])))
                for tup in dataset[token]:
                    if tup['docid'] not in doc_dict:
                        doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                    tf_document = tup['tf'] / word_count[tup['docid']]
                    doc_dict[tup['docid']][i] = np.log(1+tf_document)*idf

        # topicid: {docid: [0.009, 0.122]}

        # tfidf query topicid: [0.08, 0.9]
        #
        # docid: [0.08, 0.9] * [0.009, 0.122] = [x, y] x+y = score

        TF_IDF[topic_id] = doc_dict

    tf_idf_topic_documents = {}
    for topic_id, dic in TF_IDF.items():
        vector = tf_idf_query[topic_id]
        scores = []
        for doc_id in dic:
            scores.append((doc_id, np.dot(vector, dic[doc_id])/(np.linalg.norm(vector)*np.linalg.norm(dic[doc_id]))))

        tf_idf_topic_documents[topic_id] = np.array(scores, dtype=[('docid', np.uint32), ('score', np.float32)])
        tf_idf_topic_documents[topic_id] = np.sort(tf_idf_topic_documents[topic_id], order='score')[::-1]

    return tf_idf_topic_documents


def printable_res(sorted_tfidf: dict, run_name: str):
    string_dict = {}
    big_list = []

    for topic_id, arr in sorted_tfidf.items():
        lis = []
        counter = 0
        for tup in arr:
            if counter < 100:
                doc_id = tup['docid']
                score = '{:.2f}'.format(tup['score'])
                string = f'{topic_id} Q0 {doc_id} {counter+1} {score} {run_name}'
                lis.append(string)
                big_list.append(string)
                counter += 1

        string_dict[topic_id] = np.array(lis, dtype='<U100')

    save(os.path.join('out'), big_list)

    #return string_dict


def save(path, data: list):

    file_path = os.path.join(path, 'tfidf_title_only.txt')
    textfile = open(file_path, 'w')
    for element in data:
        textfile.write(element + '\n')
    textfile.close()


if __name__ == "__main__":
    logging.basicConfig(level=10)
    tifu = tf_idf()
    printable_res(tifu, 'run 1')
    #dataset = createindex.load(os.path.join('out', 'inverted_index.npy'))
    #print(to_print)
