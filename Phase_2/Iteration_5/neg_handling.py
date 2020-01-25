import os, datetime as dt
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import models, helper
from nltk.sentiment.util import mark_negation
from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer

def preprocess(X):
    tokenizer = TweetTokenizer()
    detokenizer = MosesDetokenizer()
    for i,value in X.iteritems():
        tokens = tokenizer.tokenize(value)
        marked_tokens = mark_negation(tokens)
        text = detokenizer.detokenize(marked_tokens)
        X.at[i] = text
    return X


data_path = helper.data_path
results_path = helper.results_path + 'Iteration5\\'
resources_path = helper.resources_path
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

X_new = preprocess(X)

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