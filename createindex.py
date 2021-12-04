"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
import pickle
import string
import sys
from time import time

import numpy
import numpy as np
import pandas
import pandas as pd
import os
import glob
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import Dict, List
import logging
import json
from multiprocessing import Pool

LOGGER = logging.getLogger('createindex')


def timed(level=15):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t_start = time()
            res = func(*args, **kwargs)
            t_end = time()
            LOGGER.log(level, f'Function {func.__name__!r} executed in {((t_end - t_start) * 1000):.0f}ms')
            return res

        return wrapper

    return decorator


@timed()
def text2tokens(text: str, pool: Pool = None) -> List[str]:
    """
    :param pool:
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """

    # Stemming takes the longest by far -> use multiprocessing to reduce times

    tokens = tokenize(text)

    if pool is None:
        # open new pool if None was parsed
        with Pool(6) as pool:
            stemmed = stem(tokens, pool)
    else:
        stemmed = stem(tokens, pool)

    return stemmed


@timed()
def tokenize(text: str) -> List[str]:
    """
    Tokenizes text and creates a List of it

    :param text: the str to tokenize
    :return: the tokenized list
    """

    sw = set(stopwords.words('english'))

    start_time = time()

    LOGGER.debug(f"Tokenizing: {text[:20]}")

    init_time = time()
    # LOGGER.debug(f"Init time: {(init_time - start_time):.6f}")

    # remove unicode chars
    text_clean = re.sub(r'[^\x00-\x7F]+', ' ', text)
    LOGGER.debug(text_clean)

    # lowercase
    text_clean = text_clean.lower()
    LOGGER.debug(text_clean)

    # remove punct
    text_clean = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text_clean)
    LOGGER.debug(text_clean)

    # remove multi space
    text_clean = re.sub(r'\s+', ' ', text_clean)
    LOGGER.debug(text_clean)

    regex_time = time()
    # LOGGER.debug(f"Regex time: {(regex_time - init_time):.6f}")

    # filter stopwords and empty string
    tokens = [w for w in text_clean.split(" ") if (w not in sw and w != "")]

    return tokens


@timed()
def stem(tokens: List[str], pool: Pool) -> List[str]:
    ps = PorterStemmer()

    text_stemmed = pool.map(ps.stem, tokens)
    LOGGER.debug(text_stemmed)

    return text_stemmed


@timed()
def add_to_inverted_index(inv_i: pd.Series, tokens: List[str], doc_id: int):
    LOGGER.debug(f"Adding to inverted index")

    for token in tokens:
        if token not in inv_i.keys():
            inv_i[token] = set()

        inv_i[token].add(doc_id)


@timed()
def create_index(tokens: List[str], doc: np.uint16):
    LOGGER.debug(f"Creating index for tokens")

    # tokens.sort()

    rev_index_doc = {}
    for i in range(len(tokens)):
        token = tokens[i]

        if token not in rev_index_doc:
            rev_index_doc[token] = []

        rev_index_doc[tokens[i]].append(i)

    for token, val in rev_index_doc.items():
        rev_index_doc[token] = len(val)
        # rev_index_doc[token] = np.uint16(val)

    # tokens_series = pd.Series(tokens)
    # cnt = pd.Series(range(0, len(tokens)), dtype=np.uint16)
    #
    # tokens_df = pd.DataFrame({'cnt': cnt, 'tokens': tokens_series})
    # grouped = tokens_df.groupby('tokens', group_keys=False)
    #
    # # grouped = tokens_series.groupby(tokens_series)
    # indices = grouped.indices
    #
    # indices_series = pd.Series(indices)
    # indices_series = pd.concat([indices_series], keys=[doc])
    # indices_series = indices_series.swaplevel()
    # indices_series = indices_series.apply(lambda x: x.astype(np.uint16))
    #
    # size = sys.getsizeof(test)
    # for val in test:
    #     size += sys.getsizeof(val)

    return rev_index_doc


def calc_dict_size(d):
    d_size = sys.getsizeof(d)
    size = 0
    for token, arr in d.items():
        size += arr.nbytes

    LOGGER.warning(f"Dict size: {d_size}, Array size: {size}")


@timed()
def merge_to_index(inv_i: Dict[str, np.ndarray], articles: Dict[np.ushort, Dict[str, np.uint16]]):
    # LOGGER.info(f"Articles: length: {len(articles)} bytes: {(sys.getsizeof(articles)):,}")
    # LOGGER.info(f"Inverted index: length: {len(inv_i)} bytes: {(sys.getsizeof(inv_i)):,}")

    for doc_id, doc in articles.items():
        for token, cnt in doc.items():
            if token not in inv_i:
                inv_i[token] = np.array([], dtype=[('docid', np.uint16), ('tf', np.uint16)])

            arr: np.ndarray = inv_i[token]
            inv_i[token] = np.append(arr, np.array((doc_id, cnt), dtype=[('docid', np.uint16), ('tf', np.uint16)]))

    return inv_i


# size = 0
# for article in articles:
#     size += sys.getsizeof(article)
# LOGGER.warning(f"SIZE: {size}")
#
# articles_series = pd.concat(articles)
#
# LOGGER.info(f"Articles series: length: {len(articles_series)} bytes: {(sys.getsizeof(articles_series)):,}")
#
# return pd.concat([inv_i, articles_series])


@timed()
def save(path, data: dict):
    LOGGER.info("Saving file")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    print(len(data))
    np.save(os.path.join(path, "inverted_index.npy"), data)
    # np.savez_compressed(os.path.join(path, "inverted_index.npy"), data)


@timed()
def load(path) -> {}:
    LOGGER.info("Loading File")

    data = np.load(path, allow_pickle=True)['arr_0'][()]

    return data


@timed(20)
def process_data(data_path):
    # load wiki files
    LOGGER.info(f"Loading files for path: {data_path}")

    time_start = time()
    # inv_i = pd.Series(dtype=np.uint16)
    inv_i = {}
    documents = 0

    with Pool(processes=6) as pool:
        for file_path in glob.iglob(os.path.join(data_path, "wikipedia articles", "1.xml")):
            time_globbing = time()
            LOGGER.info(f"Loading file: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')

                articles = {}

                for article in soup.find_all('article', recursive=False):
                    doc_id = np.uint32(article.find('header', recursive=False).find('id', recursive=False).text)
                    bdy = article.find('bdy', recursive=False)

                    if not (bdy is None or doc_id is None):
                        tokens = text2tokens(bdy.text, pool)
                        doc_index = create_index(tokens, doc_id)

                        articles[doc_id] = doc_index

                inv_i = merge_to_index(inv_i, articles)
                print(inv_i)
                # LOGGER.info(f"Articles: length: {len(articles)} bytes: {(sys.getsizeof(articles)):,}")
                # LOGGER.info(f"Inverted index: length: {len(inv_i)} bytes: {(sys.getsizeof(inv_i)):,}")
                calc_dict_size(inv_i)

            time_file_loaded = time()
            LOGGER.info(
                f"File loading time: {(time_file_loaded - time_globbing):.4f} since start: {(time_file_loaded - time_start):.4f}")

    # Sort by docid for efficient data retrieval (O(log(n)) instead of O(n))
    for token, arr in inv_i.items():
        # print(token)
        # print(arr)
        arr.sort(order='docid')
        # print(arr)

    save(os.path.join("..", "out"), inv_i)


if __name__ == "__main__":
    logging.basicConfig(level=20)
    logging.addLevelName(15, "TIMING")
    # replace with individual path to dataset
    LOGGER.debug("Starting index creation...")
    process_data(os.path.join("..", "..", "dataset"))

    # d = load(os.path.join("..", "out", "inverted_index.npy.npz"))
    # print(d)

    # with Pool(6) as pool:
    #     text2tokens("I have an ssd and I\n\t\t\t like it. This is good! I like it.", pool)
