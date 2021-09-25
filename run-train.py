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


'''
import pandas
import numpy as np
import sklearn
import nltk
import argparse
import re

# parse arguments
def add_args(parser):
    parser.add_argument('--data_dir', default='A2', type=str, help='path to directory containing training.csv')
    parser.add_argument('--model_dir', default='A2', type=str, help='path to directory where model will be saved')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COL 772 Assignment 2 | 2018EE10957')
    parser = add_args(parser)
    args = parser.parse_args()
    



    # # remove the following lines before final submission
    # python_cmd = "python run_checker.py --ground_truth_path assignment_1_data/output.json \
    #     --solution_path " + args.solution_path 
    # if args.debug:
    #     python_cmd += " --debug"
    
    # subprocess.call(python_cmd, shell=True) 

    # if args.predict != None:
    #     print(find_output_token(args.predict))