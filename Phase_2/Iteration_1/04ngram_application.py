import os, time, datetime as dt
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import models, helper

data_path = helper.data_path
results_path = helper.results_path + 'Iteration1\\02Ngram\\'
base_file = '_model_results.txt'

# import the best performing dataset
data = pd.read_csv(data_path + 'dataset-lex.csv').dropna()
data2class = data[data.target_names != 'neutral']

classifier_names = models.classifier_names
classifiers = models.classifiers
n_feats = models.n_feats

# declare evaluation metrics to be calculated
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# initialise TF-IDF vectorizers
tfidfs = {
    'BiGram': TfidfVectorizer(encoding='utf-8', ngram_range=(2, 2), stop_words='english'),
    'Combi': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 2), stop_words='english')
}

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
        scores = helper.perform_cv(pipe, data2class['data'], data2class['target'], scoring)
        # output averaged results
        helper.write_avg_results(scores, classifier_names[i], f'{basepath}{key}{base_file}')
        results[classifier_names[i]] = pd.DataFrame.from_dict(scores)
        print(f'End Time: {dt.datetime.now()}')
    # output full cross-validation results
    helper.output_cv_results(results, basepath)