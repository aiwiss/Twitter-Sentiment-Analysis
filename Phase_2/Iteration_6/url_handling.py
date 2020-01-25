import os, datetime as dt
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import models, helper
from bs4 import BeautifulSoup as bs
from urllib.request import Request, urlopen
import urllib.error as error
import http
import socket
from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer

# helper function to ensure that all characters in the URL are ASCII (legal)
def is_ascii(token):
    try:
        token.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True

# pre-process the data using URLs
def preprocess(X, data_path):
    tokenizer = TweetTokenizer()
    detokenizer = MosesDetokenizer()
    X_new = X.copy()
    edited = 0
    print(f'Start: {dt.datetime.now()}')
    # open log file
    with open(f'{data_path}P2I6log.txt', 'w', encoding='utf-8', buffering=1) as f:
        for idx,value in X_new.iteritems():
            # print current progress
            if idx % 10000 == 0:
                print(f'Current index: {idx} \nEdited: {edited} \nTime: {dt.datetime.now()}\n')
            # handle cases when URL points to Internet radios
            if 'http://' or 'https://' in value:
                if 'radio' in value:
                    continue
                tokens = tokenizer.tokenize(value)
                for i,t in enumerate(tokens):
                    if t.startswith('http://') or t.startswith('https://'):
                        if is_ascii(t):
                            try:
                                f.write(f'Trying :{t} || i: {idx} ||')
                                # create request and get html page response
                                req = Request(t, headers={'User-Agent': 'Mozilla/5.0'}) # handling website bot prevention
                                page = urlopen(req, timeout=10)
                                if not page:
                                    f.write('error\n')
                                soup = bs(page, 'html.parser')
                                # find the header within returned html and extract its text
                                if soup.find('h1'):
                                    h1text = soup.find('h1').text
                                    # replace the URL with its html page heading content
                                    tokens[i] = h1text
                                    edited += 1
                                    f.write('edited\n')
                                else:
                                    f.write('not-edited\n')
                            # handle various errors which occur when using not always the correct URLs
                            except error.HTTPError:
                                continue
                            except error.URLError:
                                continue
                            except http.client.HTTPException:
                                continue
                            except ConnectionResetError:
                                continue
                            except socket.timeout:
                                continue
                text = detokenizer.detokenize(tokens)
                X_new.at[idx] = text
    print(f'End: {dt.datetime.now()}')
    return X_new

data_path = helper.data_path
results_path = helper.results_path + 'Iteration6\\'
base_file = '_model_results.txt'

# import the best performing dataset
data = pd.read_csv(data_path + 'dataset-lex.csv').dropna()
data2class = data[data.target_names != 'neutral']

# split the data
X = data2class['data']
y = data2class['target']

# initialise classifiers
classifier_names = models.classifier_names
classifiers = models.classifiers
n_feats = models.n_feats

X_new = preprocess(X, data_path)

with open(f'{data_path}P2I6Preprocessed_tweets.csv', 'w', encoding='utf-8') as f:
    for i, val in X_new.iteritems():
        f.write(val+'\n')

# define scoring metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# initialise vectorizers
tfidfs = {
    'UniGram': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english'),
    'BiGram': TfidfVectorizer(encoding='utf-8', ngram_range=(2, 2), stop_words='english'),
    'Combi': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 2), stop_words='english')
}

# perform cross-validation
for key,value in tfidfs.items():
    tfidf = value
    results = {}
    basepath = f'{results_path}{key}\\'
    for i, clf in enumerate(classifiers):
        print(f'\nClassifier: {classifier_names[i]} \nStart Time: {dt.datetime.now()}\n')
        # initialise feature selector with k value needed
        selector = SelectKBest(score_func=chi2, k=n_feats[i])
        # intiialise the pipeline to be used for cross-validation
        pipe = make_pipeline(tfidf, selector, clf)
        scores = helper.perform_cv(pipe, X_new, y, scoring)
        # output averaged results
        helper.write_avg_results(scores, classifier_names[i], f'{basepath}{key}{base_file}')
        results[classifier_names[i]] = pd.DataFrame.from_dict(scores)
        print(f'End Time: {dt.datetime.now()}')
    # output full cross-validation results
    helper.output_cv_results(results, basepath)