'''
COL772 A2 - Shubham Mittal 2018EE10957

Sentiment Mining
----------------
    Building a tweet-sentiment categorization system using classical ML algorithms and hand-engineered features

    Preprocessing techniques
    ------------------------
        websites: www.rooms-istria.com, FetchMP3.com!, 09TOOT.BLOGSPOT.COM

    Features
    --------
        Get data into the format that sklearn requires using nltk (Use nltk only to clean the data)
        If after pre-processing, empty sentence remains then its corresponding feature (from tfidf) came out to be [0 0 ... 0]
    
    Machine Learning Models
    -----------------------

    Insights from the data
    ----------------------
        CountVectorizer(ngram_range=(1,2), min_df=0.1) gives only 11 features and when stopwords removed then none. 
            -==> stop words constitute significant part of the tweets and thus can't remove them

    References
    ----------
        https://stackoverflow.com/questions/9084237/what-is-amp-used-for
        https://medium.com/analytics-vidhya/introduction-bd62190f6acd
'''
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
    lemmatizer = nltk.WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]
    return " ".join(tokens)
    
def preprocess(x_df):
    x_df = x_df.apply(lambda sen : re.sub(r"@\S+" ,"", sen)) # remove mentions/emails
    x_df = x_df.apply(lambda sen : re.sub(r"http(s?)\S+" ,"", sen, flags=re.IGNORECASE)) # remove hyperlinks
    x_df = x_df.apply(lambda sen : re.sub(r"(www\.\S+)|(\S+\.com[^a-zA-Z\n\r])" ,"", sen, flags=re.IGNORECASE)) # remove websites
    x_df = x_df.apply(lambda sen : re.sub(r"&\w+;" ,"", sen)) # remove html tags
    x_df = x_df.apply(lambda sen : re.sub(r"(.)\1{2,}", r"\1\1", sen)) # replace char repeated > 2 times
    x_df = x_df.apply(lambda sen : stem_lemmatize(sen)) # stemming, lemmatize
    x_df = x_df.apply(lambda sen : re.sub(r"\d+", "", sen)) # remove numbers
    # x_df = x_df.apply(lambda sen : sen.translate(str.maketrans(dict.fromkeys(string.punctuation)))) # remove punctuations
    return x_df

def main(args):
    # reading training data
    training_data = pandas.read_csv(os.path.join(args.data_dir, 'train.csv'), header=None, encoding='latin-1')
    training_data[0] = training_data[0].replace(4,1)
    training_data = sklearn.utils.shuffle(training_data, random_state=18).reset_index(drop=True)
    # print("training data info\n", training_data.info())

    print("preprocessing and word2vec")
    training_data[1] = preprocess(training_data[1])
    _vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
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