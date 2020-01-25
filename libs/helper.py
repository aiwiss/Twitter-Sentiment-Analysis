import os, numpy as np
from sklearn.model_selection import cross_validate
from collections import defaultdict
from nltk.corpus import wordnet

# base paths
base_path = os.environ['OneDrive'] + '\\Sentiment Analysis\\'
resources_path = base_path + 'Code\\resources\\'
data_path = base_path + 'Data\\'
results_path = base_path + 'Data\\Results\\'

# perform 10-fold cross-validation 3 times and return full results
def perform_cv(pipeline, X, y, scoring):
    scores = defaultdict(list)
    for val in range(3):
        cvscores = cross_validate(pipeline, X, y, cv=10, scoring=scoring, n_jobs=-1)
        for k,v in cvscores.items():
            scores[k].extend(v)
    return scores

# write averaged standard metric results to file for initial evaluation
def write_avg_results(scores, clf_name, filepath):
    with open(filepath, 'a', buffering=1) as output_results:
        accuracy = scores['test_accuracy']
        precision = scores['test_precision_macro']
        recall = scores['test_recall_macro']
        f1_score = scores['test_f1_macro']
        output_results.write(f'Classifier: {clf_name}\n\n')
        output_results.write(f'Average accuracy: {"{0:.3f}".format(np.mean(accuracy))}\n')
        output_results.write(f'Average precision: {"{0:.3f}".format(np.mean(precision))}\n')
        output_results.write(f'Average recall: {"{0:.3f}".format(np.mean(recall))}\n')
        output_results.write(f'Average f1_score: {"{0:.3f}".format(np.mean(f1_score))}\n\n')

# write full model performance results to .csv file so it can be read later as a dictionary
def output_cv_results(results, filepath):
    for k,v in results.items():
        v.to_csv(f'{filepath}{k}.csv', index=False)

# convert PENN word tags to simpler - wordnet representation
def penn_to_wordnet_tag(penn_tag):
    if penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    return None