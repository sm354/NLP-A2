# COL772 A2 - Shubham Mittal 2018EE10957
import pandas
import numpy as np
import sklearn
import nltk
import argparse
import re

# parse arguments
def add_args(parser):
    parser.add_argument('--input', default='A2', type=str, help='path to input file (not necessarily csv)')
    parser.add_argument('--output', default='A2', type=str, help='path to output file')
    parser.add_argument('--model_dir', default='A2', type=str, help='path to directory containing saved model')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COL 772 Assignment 2 | 2018EE10957')
    parser = add_args(parser)
    args = parser.parse_args()
    