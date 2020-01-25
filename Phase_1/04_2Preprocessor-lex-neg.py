# Data pre-processor for lexicon-based data annotation with handled negation

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer
import re
import csv
import pandas as pd
import helper

data_path = helper.data_path
tokenizer = TweetTokenizer()
detokenizer = MosesDetokenizer()

def clean_text(text_tokens):
    for i, t in enumerate(text_tokens):
        # replace user mentions with token USER
        if t.startswith('@'):
            text_tokens[i] = 'user'
         # remove retweet tokens
        if t == 'rt':
            del text_tokens[i]

    return text_tokens

def preprocess_tweet(text):
    # handle cases when there is a space between hashtag or user mention
    tweet = re.sub('# ', '#', text)
    tweet = re.sub('@ ', '@', tweet)
    # normalise the data
    tweet = tweet.lower()
    text_tokens = tokenizer.tokenize(tweet)
    text_tokens = clean_text(text_tokens)
    
    return text_tokens

with open(data_path + 'BrexitDatasetRawNoDups.csv', encoding='utf-8') as csv_input, open(data_path + 'PreprocessedBrexitDataset-lex-neg.csv', 'a', encoding='utf-8', newline='') as csv_output:
    reader = csv.DictReader(csv_input)
    writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['date', 'data'])
    # pre-process and output the dataset to be used for lexicon-based data annotation with handled negation
    for tweet in reader:
        tokens = preprocess_tweet(tweet['Tweet'])
        text = detokenizer.detokenize(tokens)
        writer.writerow([tweet['Date'], text])

