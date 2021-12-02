"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
import string
from time import time

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
def text2tokens(text: str) -> List[str]:
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """

    # Stemming takes the longest by far

    tokens = tokenize(text)
    stemmed = stem(tokens)

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
    LOGGER.info(f"Init time: {(init_time - start_time):.6f}")

    # remove unicode chars
    text_clean = re.sub(r'[^\x00-\x7F]+', ' ', text)
    LOGGER.debug(text_clean)

    # remove new lines and tabs
    text_clean = re.sub(r'(\t\n(/n)(/t))+', ' ', text_clean)

    # lowercase
    text_clean = text_clean.lower()
    LOGGER.debug(text_clean)

    # remove punct
    text_clean = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text_clean)
    LOGGER.debug(text_clean)

    # remove multi space
    text_clean = re.sub(r'\s{+}', ' ', text_clean)
    LOGGER.debug(text_clean)

    regex_time = time()
    LOGGER.info(f"Regex time: {(regex_time - init_time):.6f}")

    # filter stopwords and empty string
    tokens = [w for w in text_clean.split(" ") if (w not in sw and w != "")]

    return tokens


@timed()
def stem(tokens: List[str]) -> List[str]:
    ps = PorterStemmer()

    text_stemmed = [ps.stem(w) for w in tokens]
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
def create_index(tokens: List[str], doc: int):
    LOGGER.debug(f"Creating index for tokens")

    tokens_series = pd.Series(tokens)

    grouped = tokens_series.groupby(tokens_series)
    indices = grouped.indices

    indices_series = pd.Series(indices)
    indices_series = pd.concat([indices_series], keys=[doc])
    indices_series = indices_series.swaplevel()

    return indices_series


@timed()
def merge_to_index(inv_i: pd.Series, doc_i: pd.Series):
    pass


@timed()
def save(path, data: pd.Series):
    LOGGER.debug("Saving file")
    #
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    # data.to_csv(os.path.join(path, "inverted_index.csv"))
    # data.to_json(os.path.join(path, "inverted_index.json"))
    #
    # with open(path, "w") as f:
    #     json.dump(data, f, indent=3, sort_keys=True)

    data = data.sort_index()

    # TODO:

    print(data)

    d = {}

    # for token in data.index.get_level_values(0):
    #     # s = {}
    #     # for doc_id in data[token]:
    #     #     s[doc_id] =
    #
    #     d[token] = data[token]
    print(d)


@timed(20)
def load_files(data_path):
    # load wiki files
    LOGGER.info(f"Loading files for path: {data_path}")

    time_start = time()
    inv_i = pd.Series()

    for file_path in glob.iglob(os.path.join(data_path, "wikipedia articles", "1.xml")):
        LOGGER.info(f"Loading file: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

            time_file_loaded = time()
            LOGGER.info(f"File loading time: {(time_file_loaded - time_start):.4f}")

            articles = []

            for article in soup.find_all('article', recursive=False):
                doc_id = int(article.find('header', recursive=False).find('id', recursive=False).text)
                bdy = article.find('bdy', recursive=False)

                if not (bdy is None or doc_id is None):
                    tokens = text2tokens(bdy.text)
                    doc_index = create_index(tokens, doc_id)

                    articles.append(doc_index)

                    # inv_i = pd.concat([pd.concat([doc_index], keys=[doc_id]), inv_i])

                    # print(doc_index)
                    # merge_to_index(inv_i, doc_index)
                    # add_to_inverted_index(inv_i, tokens, int(doc_id.text))

            inv_i = pd.concat([pd.concat(articles), inv_i])

    save(os.path.join("..", "out"), inv_i)

    # inv_i_list = {}
    # for key, value in inv_i.items():
    #     inv_i_list[key] = list(value)

    # save_to_json(os.path.join("..", "out", "inverted_index.json"), inv_i_list)
    # save_to_json(os.path.join("..", "out", "inverted_index_keys.json"), list(inv_i.keys()))


if __name__ == "__main__":
    logging.basicConfig(level=15)
    logging.addLevelName(15, "TIMING")
    # replace with individual path to dataset
    LOGGER.info("Starting index creation...")
    load_files(os.path.join("..", "..", "dataset"))

    # text2tokens("I have an ssd and I like it. This is good! I like it.")
