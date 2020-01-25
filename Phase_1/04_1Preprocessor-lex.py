# Data pre-processor for lexicon-based data annotation

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

# remove unnecessary data from tweets
def clean_text(text_tokens):
    stop_words = set(stopwords.words('english'))
    # remove stop words
    filtered_tokens = [t for t in text_tokens if not t in stop_words]
    for i, t in enumerate(filtered_tokens):
        # replace user mentions with token USER
        if t.startswith('@'):
            filtered_tokens[i] = 'USER'
        # remove retweet tokens
        if t == 'rt':
            del filtered_tokens[i]

    return filtered_tokens

def preprocess_tweet(text):
    # handle cases when there is a space between hashtag or user mention
    tweet = re.sub('# ', '#', text)
    tweet = re.sub('@ ', '@', tweet)
    # normalise the data
    tweet = tweet.lower()
    text_tokens = tokenizer.tokenize(tweet)
    text_tokens = clean_text(text_tokens)
    
    return text_tokens

# remove duplicate data which is usually produced by bots so check for the same tweet text
def remove_duplicates():
    raw_data = pd.read_csv(data_path + 'BrexitDatasetRaw.csv')
    clean_data = raw_data.drop_duplicates(subset=['Tweet'])
    clean_data.to_csv(data_path + 'BrexitDatasetRawNoDups.csv')

remove_duplicates()

with open(data_path + 'BrexitDatasetRawNoDups.csv', encoding='utf-8') as csv_input, open(data_path + 'PreprocessedBrexitDataset-lex.csv', 'a', encoding='utf-8', newline='') as csv_output:
    reader = csv.DictReader(csv_input)
    writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['date', 'data'])
    # pre-process and output the dataset to be used for lexicon-based data annotation
    for tweet in reader:
        tokens = preprocess_tweet(tweet['Tweet'])
        text = detokenizer.detokenize(tokens)
        writer.writerow([tweet['Date'], text])

