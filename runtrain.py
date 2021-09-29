'''
COL772 A2 - Shubham Mittal 2018EE10957

Sentiment Mining
----------------
    Building a tweet-sentiment categorization system using classical ML algorithms and hand-engineered features

    Preprocessing techniques
    ------------------------

    Features
    --------

    Machine Learning Models
    -----------------------

    Insights from the data
    ----------------------
        CountVectorizer(ngram_range=(1,2), min_df=0.1) gives only 11 features and when stopwords removed then none. 
            -==> stop words constitute significant part of the tweets and thus can't remove them

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
    tokens = word_tokenize(sen)

    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]
    return " ".join(tokens)
    
def preprocess(x_df):
    x_df = x_df.apply(lambda sen : re.sub(r"@\S+" ,"", sen)) # remove mentions
    x_df = x_df.apply(lambda sen : re.sub(r"http\S+" ,"", sen)) # remove hyperlinks
    x_df = x_df.apply(lambda sen : sen.translate(str.maketrans(dict.fromkeys(string.punctuation)))) # remove punctuations
    # transform loooooove to loove
    # x_df = x_df.apply(lambda sen : stem_lemmatize(sen)) # stemming, lemmatize
    return x_df

def main(args):
    # reading training data
    training_data = pandas.read_csv(os.path.join(args.data_dir, 'train.csv'), header=None, encoding='latin-1')
    training_data[0] - training_data[0].replace(4,1)
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
    



    # # remove the following lines before final submission
    # python_cmd = "python run_checker.py --ground_truth_path assignment_1_data/output.json \
    #     --solution_path " + args.solution_path 
    # if args.debug:
    #     python_cmd += " --debug"
    
    # subprocess.call(python_cmd, shell=True) 

    # if args.predict != None:
    #     print(find_output_token(args.predict))