import os, time, datetime as dt
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import models, helper

data_path = helper.data_path
results_path = helper.results_path + 'Iteration1\\'

# read annotated dataset
data = pd.read_csv(data_path + 'dataset-lex.csv')

data2class = data.dropna()
# extract 2-class dataset
data2class = data[data.target_names != 'neutral']

# get classifiers
n_feats = models.n_feats
classifier_names = models.classifier_names
classifiers = models.classifiers

# declare parameter grid
parameters = [
    {'n_neighbors':(1, 3, 5)},
    {'max_depth':(3, 5, 7),'min_samples_split':(2,3,4)},
    {'max_depth':(80,100,120),'n_estimators':(10,100,150)},
    {'alpha':(0.01,0.1,1)},
    {'alpha':(0.01,0.1,1)},
    {'C':(1,3,5)},
    {'alpha':(0.0001, 0.001, 0.1),'hidden_layer_sizes':[(10,10),(100,100),(150,150)]}
]

# initialiseTF-IDF vectorizer
tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english')
X_tfidf = tfidf.fit_transform(data2class['data'], data2class['target'])
y = data2class['target'].values.reshape(-1,1)


with open(f'{results_path}parameter_gs_results.txt', 'w', buffering=1) as output_results:
    for i, clf in enumerate(classifiers):
        # initialise feature selector with k value needed
        selector = SelectKBest(score_func=chi2, k=n_feats[i]).fit(X_tfidf,y)
        # select k-best features
        X_tfidf_new = selector.transform(X_tfidf)
        start = time.time()
        output_results.write('Classifier: ' + classifier_names[i] + '\n')
        print('\nClassifier: ', classifier_names[i])
        print('Start Time: ', dt.datetime.now())
        # initialise GridSearch cross-validation
        gs = GridSearchCV(clf, parameters[i], cv=5, scoring='accuracy', n_jobs=-1)
        # perform GridSearch and get results for different hyperparameter combinations
        gs.fit(X_tfidf_new,data2class['target'])
        
        output_results.write(f'Results: \n')
        
        for key,value in gs.cv_results_.items():
            output_results.write(f'{key}: {value} \n')

        output_results.write(f'\nBest parameter: {gs.best_params_} \n')
        output_results.write(f'Best score: {gs.best_score_} \n\n')
        
        end = time.time()
        if end - start > 28800:
            break

        print('End Time: ', dt.datetime.now())
