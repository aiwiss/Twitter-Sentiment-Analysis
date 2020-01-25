# Feature selection step code

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
results_path = helper.results_path + 'Iteration1\\'

# declare the k value tests
kbest_tests = [600,1000,1500,2500]

# load dataset
data = pd.read_csv(data_path + 'dataset-lex.csv')

# split into 2-class and 3-class
data3class = data.dropna()
data2class = data[data.target_names != 'neutral']

# initialise the list of classifiers
classifier_names = models.classifier_names
classifiers = models.classifiers

# initialise tf-idf vectorizers
tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english')
tfidf2class = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english')
tfidf3class = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english')

# output kbest feature results
with open(results_path + 'feature_selection_results.txt', 'a', buffering=1) as output_results:
    output_results.write('================= K Best Resuls =================\n')
    for i, clf in enumerate(classifiers):
        output_results.write('\nClassifier: ' + classifier_names[i] + '\n\n')
        print('Classifier: ', classifier_names[i])
        print('Start Time: ', dt.datetime.now())
        scores = []
        for k in kbest_tests:
            start = time.time()
            output_results.write('\nK: ' + str(k) + '\n')
            # initialise feature selector with k value needed
            selector = SelectKBest(score_func=chi2, k=k)
            # intiialise the pipeline to be used for cross-validation
            pipe = make_pipeline(tfidf, selector, clf)
            # get feature selection results using 2-class and 3-class datasets
            scores2class = cross_val_score(pipe, data2class['data'], data2class['target'], scoring='accuracy', cv=10, n_jobs=-1)
            scores3class = cross_val_score(pipe, data3class['data'], data3class['target'], scoring='accuracy', cv=10, n_jobs=-1)
            output_results.write('2-class average accuracy: ' + str("{0:.3f}".format(np.mean(scores2class))) + '\n')
            output_results.write('3-class average accuracy: ' + str("{0:.3f}".format(np.mean(scores3class))) + '\n')
            end = time.time()
            if end - start > 28800:
                break
        print('End Time: ', dt.datetime.now())
    output_results.write('\n\n================= RF based feature importance results =================\n')
    
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

    for i, clf in enumerate(classifiers):
        output_results.write('Classifier: ' + classifier_names[i] + '\n')
        print('\nClassifier: ', classifier_names[i])
        print('Start Time: ', dt.datetime.now())

        # intialise the pipelines for both datasets and use it for cross-validation
        pipe2 = make_pipeline(tfidf, SelectFromModel(rf_clf, threshold=rf_2class_min_weight), clf)
        pipe3 = make_pipeline(tfidf, SelectFromModel(rf_clf, threshold=rf_3class_min_weight), clf)
        scores2class = cross_val_score(pipe2, data2class['data'], data2class['target'], scoring='accuracy', cv=10, n_jobs=-1)
        scores3class = cross_val_score(pipe3, data3class['data'], data3class['target'], scoring='accuracy', cv=10, n_jobs=-1)
        # output averaged results
        output_results.write('2-class average accuracy: ' + str("{0:.3f}".format(np.mean(scores2class))) + '\n')
        output_results.write('3-class average accuracy: ' + str("{0:.3f}".format(np.mean(scores3class))) + '\n')
        
        print('End Time: ', dt.datetime.now())


    

    