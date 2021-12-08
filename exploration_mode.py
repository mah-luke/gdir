"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked list of documents is returned.
Make sure that the user can switch between TF-IDF and BM25 scoring functions.
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
from createindex import *


logging.basicConfig(level=15)
LOGGER = logging.getLogger('exploration')


class Exploration:

    def __init__(self, dataset: dict[str, np.ndarray] = None, path_dict=None):
        self.dataset = dataset or createindex.load(os.path.join('..', 'out', 'inverted_index.npz'))
        self.path_dict = path_dict or createindex.load(os.path.join('..', 'out', 'file_doc_index.npz'))

    @timed()
    def tf_idf_simple(self, query: str):
        tokens = createindex.text2tokens(query)
        word_count = {}

        # count words per document
        for token, arr in self.dataset.items():
            for tup in arr:
                if tup['docid'] not in word_count:
                    word_count[tup['docid']] = tup['tf']
                else:
                    word_count[tup['docid']] += tup['tf']

        N = len(word_count)

        doc_dict = {}
        for i in range(len(tokens)):
            token = tokens[i]
            if token in self.dataset:
                idf = np.log(N / len(self.dataset[token]))
                for tup in self.dataset[token]:
                    if tup['docid'] not in doc_dict:
                        doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                    tf_doc = tup['tf']
                    doc_dict[tup['docid']][i] = np.log(1 + tf_doc) * idf

        scores = []
        for docid in doc_dict:
            scores.append((docid, np.sum(doc_dict[docid])))

        tf_idf_documents = np.array(scores, dtype=[('docid', np.uint32), ('score', np.float32)])
        tf_idf_documents = np.sort(tf_idf_documents, order='score')[::-1]

        return tf_idf_documents

    @timed()
    def tf_idf_cosine(self, query: str):
        tokens = createindex.text2tokens(query)
        word_count = {}

        # count words per document
        for token, arr in self.dataset.items():
            for tup in arr:
                if tup['docid'] not in word_count:
                    word_count[tup['docid']] = tup['tf']
                else:
                    word_count[tup['docid']] += tup['tf']

        N = len(word_count)

        # tfidf for query
        scores_query = []
        for i in range(len(tokens)):
            token = tokens[i]
            if token in self.dataset:
                idf = np.log(N/len(self.dataset[token]))
                tf_query = collections.Counter(tokens)[token]
                scores_query.append(np.log(1+tf_query)*idf)
            else:
                scores_query.append(0)

        tfidf_query = np.array(scores_query, dtype=np.float32)

        # tfidf for documents
        doc_dict = {}
        for token in tokens:
            for i in range(len(tokens)):
                token = tokens[i]
                if token in self.dataset:
                    idf = np.log(N / len(self.dataset[token]))
                    for tup in self.dataset[token]:
                        if tup['docid'] not in doc_dict:
                            doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                        tf_doc = tup['tf']
                        doc_dict[tup['docid']][i] = np.log(1 + tf_doc) * idf

        scores = []
        for docid, arr in doc_dict.items():
            scores.append((docid, np.dot(tfidf_query, arr)/(np.lianlg.norm(tfidf_query)*np.linalg.norm(arr))))

        tf_idf_documents = np.array(scores, dtype=[('docid', np.uint32), ('score', np.float32)])
        tf_idf_documents = np.sort(tf_idf_documents, order='score')[::-1]

        return tf_idf_documents


    @timed()
    def bm25(self, query: str, k1: float, b: float):
        tokens = createindex.text2tokens(query)
        word_count = {}

        # count words per document
        for token, arr in self.dataset.items():
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

        doc_dict = {}
        for i in range(len(tokens)):
            token = tokens[i]
            if token in self.dataset:
                for tup in self.dataset[token]:
                    if tup['docid'] not in doc_dict:
                        doc_dict[tup['docid']] = np.zeros(len(tokens), dtype=np.float32)

                    tf_doc = tup['tf']
                    doc_dict[tup['docid']][i] = tf_doc

        scores = []
        for docid in doc_dict:
            add = 0
            for i in range(len(tokens)):
                token = tokens[i]
                if token in self.dataset:
                    const = np.log(N/len(self.dataset[token]))
                    numerator = (k1+1)*doc_dict[docid][i]
                    denominator = k1*((1-b)+b*(word_count[docid]/avg_doc_length))+doc_dict[docid][i]
                    op = const*(numerator/denominator)
                    add += op
            scores.append((docid, add))

        bm25_ranking = np.array(scores, dtype=[('docid', np.uint32), ('score', np.float32)])
        bm25_ranking = np.sort(bm25_ranking, order='score')[::-1]

        return bm25_ranking

    def print_results(self, sorted_list):
        big_list = []

        for tup in sorted_list[:10]:
            with open(self.find_path(tup['docid']), 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                for article in soup.find_all('article', recursive=False):
                    header = article.find('header', recursive=False)
                    cur_id = np.uint32(header.find('id', recursive=False).text)
                    if cur_id == tup['docid']:
                        title = header.find('title').text
                        bdy = article.find('bdy').text

                        content = f"SCORING: {tup['score']}, DOCID: {tup['docid']} RESULT:\n{title}\n{bdy[:100]}\n ... CONTENT ... \n{bdy[-100:]}"
                        big_list.append(content)
                        break

        return big_list

    @timed()
    def find_path(self, docid) -> str:
        for path, docids in self.path_dict.items():
            if docid in docids:
                return path


if __name__ == "__main__":

    scoring = Exploration()
    bm = scoring.bm25('proctor compaction test', 1.25, 0.75)
    results = scoring.print_results(bm)
    for result in results:
        print(result)
