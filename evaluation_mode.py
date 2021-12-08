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
from createindex import timed

logging.basicConfig(level=10)
LOGGER = logging.getLogger('evaluation_mode')


@timed()
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


@timed()
def document_number(path: str):
    titles = []

    for file_path in glob.iglob(os.path.join(path, "wikipedia articles", "1.xml")):
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            for title in soup.find_all('title', recursive=False):
                titles.append(title)

    return len(titles)


@timed()
def tf_idf_cosine_similarity():
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
    tf(t,d) = count of t in d / number of words in d

    df(t) = number of documents that contain t

    idf(t) = log(N/df(t))

    tf-idf(t, d) = log(1+tf(t, d)) * idf(t)

    dataset = {
                t1: [(1, 1), (572, 4)]
                t2: ...
               }


    topicid: {docid: [0.009, 0.122]}

    tfidf query topicid: [0.08, 0.9]

    docid: [0.08, 0.9] * [0.009, 0.122] = [x, y] x+y = score

    """
    LOGGER.info('Ranking results with TF-IDF')
    dataset = createindex.load(os.path.join('..', 'out', 'inverted_index.npz'))
    topics = parse_topics(os.path.join('..', '..', 'GIR2021 dataset'))
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
        for i in range(len(tokens)):
            token = tokens[i]
            if token in dataset:
                idf = np.log(N / len(dataset[token]))
                #tf_query = collections.Counter(tokens)[token]/len(tokens)
                tf_query = collections.Counter(tokens)[token]
                scores.append(np.log(1+tf_query) * idf)
            else:
                scores.append(0)

        tf_idf_query[topic_id] = np.array(scores, dtype=numpy.float32)


    # Calculate tf-idf for documents
    for topic_id, tokens in topics.items():
        doc_dict = {}
        for i in range(len(tokens)):
            token = tokens[i]
            if token in dataset:
                idf = np.log(N / len(dataset[token]))
                for tup in dataset[token]:
                    if tup['docid'] not in doc_dict:
                        doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                    #tf_document = tup['tf'] / word_count[tup['docid']]
                    tf_document = tup['tf']
                    doc_dict[tup['docid']][i] = np.log(1+tf_document)*idf

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


@timed()
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
    tf(t,d) = count of t in d / number of words in d

    df(t) = number of documents that contain t

    idf(t) = log(N/df(t))

    tf-idf(t, d) = log(1+tf(t, d)) * idf(t)

    dataset = {
                t1: [(1, 1), (572, 4)]
                t2: ...
               }


    topicid: {docid: [0.009, 0.122]}

    tfidf query topicid: [0.08, 0.9]

    docid: [0.08, 0.9] * [0.009, 0.122] = [x, y] x+y = score

    """
    LOGGER.info('Ranking results with TF-IDF')
    dataset = createindex.load(os.path.join('..', 'out', 'inverted_index.npz'))
    df = {}
    topics = parse_topics(os.path.join('..', '..', 'GIR2021 dataset'))
    word_count = {}

    for token, arr in dataset.items():
        for tup in arr:
            if tup['docid'] not in word_count:
                word_count[tup['docid']] = tup['tf']
            else:
                word_count[tup['docid']] += tup['tf']

    TF_IDF = {}
    N = len(word_count)

    # Calculate tf-idf for documents
    for topic_id, tokens in topics.items():
        doc_dict = {}
        for i in range(len(tokens)):
            token = tokens[i]
            if token in dataset:
                idf = np.log(N / len(dataset[token]))
                for tup in dataset[token]:
                    if tup['docid'] not in doc_dict:
                        doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                    #tf_document = tup['tf'] / word_count[tup['docid']]
                    tf_document = tup['tf']
                    doc_dict[tup['docid']][i] = np.log(1+tf_document)*idf

        TF_IDF[topic_id] = doc_dict

    tf_idf_topic_documents = {}
    for topic_id, dic in TF_IDF.items():
        scores = []
        for doc_id in dic:
            scores.append((doc_id, np.sum(dic[doc_id])))

        tf_idf_topic_documents[topic_id] = np.array(scores, dtype=[('docid', np.uint32), ('score', np.float32)])
        tf_idf_topic_documents[topic_id] = np.sort(tf_idf_topic_documents[topic_id], order='score')[::-1]

    return tf_idf_topic_documents


@timed()
def bm25(k1: float, b: float):
    LOGGER.info('Ranking results with BM25')
    dataset = createindex.load(os.path.join('..', 'out', 'inverted_index.npz'))
    topics = parse_topics(os.path.join('..', '..', 'GIR2021 dataset'))
    word_count = {}
    avg_doc_length = 0
    TF = {}

    for token, arr in dataset.items():
        for tup in arr:
            if tup['docid'] not in word_count:
                word_count[tup['docid']] = tup['tf']
            else:
                word_count[tup['docid']] += tup['tf']

    N = len(word_count)
    tmp = []
    for wc in word_count:
        tmp.append(word_count[wc])

    avg_doc_length = np.mean(tmp)

    for topic_id, tokens in topics.items():
        doc_dict = {}
        for i in range(len(tokens)):
            token = tokens[i]
            if token in dataset:
                for tup in dataset[token]:
                    if tup['docid'] not in doc_dict:
                        doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                    # tf_document = tup['tf'] / word_count[tup['docid']]
                    tf_document = tup['tf']
                    doc_dict[tup['docid']][i] = tf_document

        TF[topic_id] = doc_dict

    bm25_record = {}
    for topic_id, tokens in topics.items():
        doc_dict = TF[topic_id]
        scores = []
        for docid in doc_dict:
            add = 0
            for i in range(len(tokens)):
                token = tokens[i]
                if token in dataset:
                    const = np.log(N / len(dataset[token]))
                    numerator = (k1+1)*TF[topic_id][docid][i]
                    denominator = k1*((1-b)+b*(word_count[docid]/avg_doc_length))+TF[topic_id][docid][i]
                    op = const*(numerator/denominator)
                    add += op
            scores.append((docid, add))

        bm25_record[topic_id] = np.array(scores, dtype=[('docid', np.uint32), ('score', np.float32)])
        bm25_record[topic_id] = np.sort(bm25_record[topic_id], order='score')[::-1]

    return bm25_record

@timed()
def printable_res(sorted_res: dict, run_name: str):
    string_dict = {}
    big_list = []

    for topic_id, arr in sorted_res.items():
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

    if run_name == 'tfidf':
        save(os.path.join('..', 'out'), big_list, 'tfidf_title_only.txt')
    elif run_name == 'bm25':
        save(os.path.join('..', 'out'), big_list, 'bm25_title_only.txt')
    elif run_name == 'tfidf_cs':
        save(os.path.join('..', 'out'), big_list, 'tfidf_cs_title_only.txt')


@timed()
def save(path, data: list, name: str):

    file_path = os.path.join(path, name)
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
    #bm = bm25(1.25, 0.75)
    #printable_res(bm, 'run 2')
