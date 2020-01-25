# Additional feature selection code to get more results according to the changed methodology

import time, datetime as dt
import numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import models
import helper

data_path = helper.data_path
results_path = helper.results_path + 'Iteration1\\additional\\'
base_file = '_model_results.txt'

k_best = models.n_feats

# load dataset
data = pd.read_csv(data_path + 'dataset-lex.csv')

# split into 2-class and 3-class
data3class = data.dropna()
data2class = data[data.target_names != 'neutral']

# initialise tf-idf vectorizers
tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english')
tfidf2class = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english')
tfidf3class = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english')

# initialise classifiers
classifier_names = models.classifier_names
classifiers = models.classifiers
n_feats = models.n_feats

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# get both, 2-class and 3-class dataset extensive performance results
data2 = data2class
for j in range(2):
    results = {}
    basepath = f'{results_path}kbest\\{i}\\'
    for i, clf in enumerate(classifiers):
        print(f'\nClassifier: {classifier_names[i]} \nStart Time: {dt.datetime.now()}\n')
        # initialise feature selector with k value needed
        selector = SelectKBest(score_func=chi2, k=n_feats[i])
        # intiialise the pipeline to be used for cross-validation
        pipe = make_pipeline(tfidf, selector, clf)
        # get feature selection results using 2-class and 3-class datasets
        scores = helper.perform_cv(pipe, data2['data'], data2['target'], scoring)
        helper.write_avg_results(scores, classifier_names[i], f'{basepath}UniGram{base_file}')
        results[classifier_names[i]] = pd.DataFrame.from_dict(scores)
        print(f'End Time: {dt.datetime.now()}')
    helper.output_cv_results(results, basepath)
    data2 = data3class

# extract unigram features from both datasets using TF-IDF vectorizer
X_tfidf2class = tfidf2class.fit_transform(data2class['data'])
X_tfidf3class = tfidf3class.fit_transform(data3class['data'])

# intialise Random Forest model to be used for feature selection
rf_clf = RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1)

# initialise 2-class RF classifier for feature ranking
rf_clf2class = RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1).fit(X_tfidf2class, data2class['target'])
df_rf_clf2class_weights = pd.DataFrame(rf_clf2class.feature_importances_, columns=['importance'])

# get the minimum importance weight so only feature having importance are selected
rf_2class_min_weight = df_rf_clf2class_weights[df_rf_clf2class_weights['importance'] > 0].min()

# initialise 3-class RF classifier for feature ranking
rf_clf3class = RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1).fit(X_tfidf3class, data3class['target'])
df_rf_clf3class_weights = pd.DataFrame(rf_clf3class.feature_importances_, columns=['importance'])

# get the minimum importance weight so only feature having importance are selected
rf_3class_min_weight = df_rf_clf3class_weights[df_rf_clf3class_weights['importance'] > 0].min()

sfms = [
    SelectFromModel(rf_clf, threshold=rf_2class_min_weight),
    SelectFromModel(rf_clf, threshold=rf_3class_min_weight)
]

data2 = data2class
for j,sfm in enumerate(sfms):
    results = {}
    basepath = f'{results_path}rf\\{j}\\'
    for i, clf in enumerate(classifiers):
        print(f'\nClassifier: {classifier_names[i]} \nStart Time: {dt.datetime.now()}\n')
        # intialise the pipeline to be used for cross-validation
        pipe = make_pipeline(tfidf, sfm, clf)
        scores = helper.perform_cv(pipe, data2['data'], data2['target'], scoring)
        # output averaged results
        helper.write_avg_results(scores, classifier_names[i], f'{basepath}UniGram{base_file}')
        results[classifier_names[i]] = pd.DataFrame.from_dict(scores)
        print(f'End Time: {dt.datetime.now()}')
    # output full cross-validation results
    helper.output_cv_results(results, basepath)
    data2 = data3class    