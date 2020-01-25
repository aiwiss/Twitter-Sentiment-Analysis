import os, time, datetime as dt
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import models, helper
from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer

# helper function to read slang words
def load_slang(text):
    result = {}
    for line in text:
        l = line.split(':')
        result[l[0]] = l[1]
    return result

# helper function to read abbreviations dictionary
def load_abbreviations(text):
    result = {}
    for line in text:
        l = line.split('@')
        result[l[0]] = l[1]
    return result

# pre-process tweets by replacing slang and abbreviations
def preprocess_tweet(tweet, slang_keys, slang_dictionary, abbr_keys, abbr_dictionary):
    tokenizer = TweetTokenizer()
    detokenizer = MosesDetokenizer()
    tokens = tokenizer.tokenize(tweet)
    for i,t in enumerate(tokens):
        # replace slang
        if t in slang_keys:
            tokens[i] = slang_dictionary[t]
        
        # replace abbreviations 
        if t.upper() in abbr_keys:
            tokens[i] = abbr_dictionary[t.upper()]
    
    new_tweet = detokenizer.detokenize(tokens)
    return new_tweet

data_path = helper.data_path
results_path = helper.results_path + 'Iteration2\\'
resources_path = helper.resources_path
base_file = '_model_results.txt'

# import the best performing dataset
data = pd.read_csv(data_path + 'dataset-lex.csv').dropna()
data2class = data[data.target_names != 'neutral']

#import slang and abbreviation dictionaries
slang = {}
abbreviations = {}
with open(resources_path + 'slang_dictionary.txt', encoding='utf-8') as f:
    text = f.read().splitlines()
    slang = load_slang(text)

with open(resources_path + 'abbreviations_dictionary.txt', encoding='utf-8') as f:
    text = f.read().splitlines()
    abbreviations = load_abbreviations(text)

classifier_names = models.classifier_names
classifiers = models.classifiers
n_feats = models.n_feats

# declare evaluation metrics to be calculated
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# initialise TF-IDF vectorizers
tfidfs = {
    'UniGram': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english'),
    'BiGram': TfidfVectorizer(encoding='utf-8', ngram_range=(2, 2), stop_words='english'),
    'Combi': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 2), stop_words='english')
}

# pre-process data
slang_keys = list(slang.keys())
abbreviations_keys = list(abbreviations.keys())
for i, row in data2class.iterrows():
    tweet = row['data']
    new_tweet = preprocess_tweet(tweet, slang_keys, slang, abbreviations_keys, abbreviations)
    data2class.at[i,'data'] = new_tweet
    
# output to file so data can be inspected
data2class.to_csv(data_path + 'P2I2PreprocessedData.csv')

X = data2class['data']
y = data2class['target']
    
# perform cross-validation
for key,value in tfidfs.items():
    tfidf = value
    results = {}
    basepath = f'{results_path}{key}\\'
    for i, clf in enumerate(classifiers):
        print(f'\nClassifier: {classifier_names[i]} \n Start Time: {dt.datetime.now()}\n')
        # initialise feature selector with k value needed
        selector = SelectKBest(score_func=chi2, k=n_feats[i])
        # intiialise the pipeline to be used for cross-validation
        pipe = make_pipeline(tfidf, selector, clf)
        scores = helper.perform_cv(pipe, X, y, scoring)
        # output averaged results
        helper.write_avg_results(scores, classifier_names[i], f'{basepath}{key}{base_file}')
        results[classifier_names[i]] = pd.DataFrame.from_dict(scores)
        print(f'End Time: {dt.datetime.now()}')
    # output full cross-validation results
    helper.output_cv_results(results, basepath)

