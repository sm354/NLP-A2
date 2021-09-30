# COL772 A2 - Shubham Mittal 2018EE10957
import pandas
import numpy as np
import sklearn
import nltk
import argparse
import re
import os
import pickle

import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

# parse arguments
def add_args(parser):
    parser.add_argument('--data_dir', default='A2', type=str, help='path to directory containing training.csv')
    parser.add_argument('--model_dir', default='A2', type=str, help='path to directory where model will be saved')
    return parser

def stem_lemmatize(sen):
    # tokens = word_tokenize(sen)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sen)

    stemmer = nltk.PorterStemmer()
    # lemmatizer = nltk.WordNetLemmatizer()
    # lemmatizer.lemmatize()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)
    
def preprocess(x_df):
    # x_df = x_df.apply(lambda sen : re.sub(r"@\S+" ,"", sen)) # remove mentions/emails
    # x_df = x_df.apply(lambda sen : re.sub(r"http(s?)\S+" ,"", sen, flags=re.IGNORECASE)) # remove hyperlinks
    # x_df = x_df.apply(lambda sen : re.sub(r"(www\.\S+)|(\S+\.com[^a-zA-Z\n\r])" ,"", sen, flags=re.IGNORECASE)) # remove websites
    # x_df = x_df.apply(lambda sen : re.sub(r"&\w+;" ,"", sen)) # remove html tags
    x_df = x_df.apply(lambda sen : re.sub(r"(.)\1{2,}", r"\1\1", sen)) # replace char repeated > 2 times
    x_df = x_df.apply(lambda sen : stem_lemmatize(sen)) # stemming, lemmatize
    # x_df = x_df.apply(lambda sen : re.sub(r"\d+", "", sen)) # remove numbers
    # x_df = x_df.apply(lambda sen : sen.translate(str.maketrans(dict.fromkeys(string.punctuation)))) # remove punctuations
    return x_df

def main(args):
    # reading training data
    training_data = pandas.read_csv(os.path.join(args.data_dir, 'training.csv'), header=None, encoding='latin-1')
    training_data[0] = training_data[0].replace(4,1)
    training_data = sklearn.utils.shuffle(training_data, random_state=18).reset_index(drop=True)
    # print("training data info\n", training_data.info())

    print("preprocessing and word2vec")
    training_data[1] = preprocess(training_data[1])
    _vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=300000)
    x_train = _vectorizer.fit_transform(training_data[1])
    pickle.dump(_vectorizer, open(os.path.join(args.model_dir, 'vectorizer'), 'wb'))

    print("training model")
    model = LogisticRegression(penalty='l2', solver='liblinear')
    model.fit(x_train, training_data[0])

    print("score on trainset:", model.score(x_train, training_data[0]))
    
    pickle.dump(model, open(os.path.join(args.model_dir, 'model'), 'wb'))
    print("model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COL 772 Assignment 2 | 2018EE10957')
    parser = add_args(parser)
    args = parser.parse_args()
    
    main(args)