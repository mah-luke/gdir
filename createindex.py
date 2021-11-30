"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
import string
from time import time

import numpy as np
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


def timed(func):
    def wrapper_function(*args, **kwargs):
        t_start = time()
        res = func(*args, **kwargs)
        t_end = time()
        LOGGER.warning(f'Function {func.__name__!r} executed in {(t_end - t_start):.4f}s')
        return res

    return wrapper_function


@timed
def text2tokens(text: str):
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """

    # Stemming takes the longest by far

    tokens = tokenize(text)
    stemmed = stem(tokens)

    return stemmed


@timed
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
    text_clean = re.sub(r'(\t\n)+', ' ', text_clean)

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


@timed
def stem(tokens: List[str]) -> List[str]:
    ps = PorterStemmer()

    text_stemmed = [ps.stem(w) for w in tokens]
    LOGGER.debug(text_stemmed)

    return text_stemmed


@timed
def add_to_inverted_index(inv_i: Dict[str, set], tokens: str, doc_id: str):
    LOGGER.debug(f"Adding to inverted index")

    for token in tokens:
        if token not in inv_i.keys():
            inv_i[token] = set()

        inv_i[token].add(doc_id)


@timed
def save_to_json(path, data):
    LOGGER.debug("Saving to json file")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "w") as f:
        json.dump(data, f, indent=3, sort_keys=True)


@timed
def load_files(data_path):
    # load wiki files
    LOGGER.info(f"Loading files for path: {data_path}")

    time_start = time()
    inv_i = {}

    for file_path in glob.iglob(os.path.join(data_path, "wikipedia articles", "*.xml")):
        LOGGER.info(f"Loading file: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

            time_file_loaded = time()
            LOGGER.info(f"File loading time: {(time_file_loaded - time_start):.4f}")

            for article in soup.find_all('article', recursive=False):
                doc_id = article.find('header', recursive=False).find('id', recursive=False)
                bdy = article.find('bdy', recursive=False)

                if bdy is not None and doc_id is not None:
                    tokens = text2tokens(bdy.text)
                    add_to_inverted_index(inv_i, tokens, int(doc_id.text))

    inv_i_list = {}
    for key, value in inv_i.items():
        inv_i_list[key] = list(value)

    save_to_json(os.path.join("..", "out", "inverted_index.json"), inv_i_list)
    save_to_json(os.path.join("..", "out", "inverted_index_keys.json"), list(inv_i.keys()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # replace with individual path to dataset
    LOGGER.info("Starting index creation...")
    load_files("../../dataset")

    # text2tokens("I have an ssd and I like it. This is good! I like it.")
