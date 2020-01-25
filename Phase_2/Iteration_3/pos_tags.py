import os, datetime as dt
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import models, helper
from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer
from nltk import pos_tag
from nltk.corpus import wordnet

# helper function to add generated pos-tag features into the csr matrix
# produced by TF-IDF vectorizer
def combine_features(X1, feature):
    X2 = feature.reset_index()
    X2 = X2.iloc[:,[1]].values
    combined = sp.hstack((X1, X2), format='csr')
    return combined

# helper function to assign pos-tags for words
def add_pos_tags(tokens):
    for i,token in enumerate(tokens):
        if token.startswith('#'):
            continue
        elif token.startswith('http'):
            continue
        else:
            # concatenate tokens with pos tags
            tagged = pos_tag([token])
            tag = helper.penn_to_wordnet_tag(tagged[0][1])
            if tag != None:
                tokens[i] = f'{token}_{tag}'
    return tokens

# function to count the pos-tag occurences needed for feature engineering
def count_pos_tags(tokens):
    nounCount = 0
    adjCount = 0
    advCount = 0
    verbCount = 0
    for t in tokens:
        if '_n' in t:
            nounCount += 1
        elif '_a' in t:
            adjCount += 1
        elif '_r' in t:
            advCount += 1
        elif '_v' in t:
            verbCount += 1
    return [nounCount, adjCount, advCount, verbCount]

def preprocess(data2class):
    # add columns to DataFrame for features
    data2class['nounCount'] = ''
    data2class['adjCount'] = ''
    data2class['advCount'] = ''
    data2class['verbCount'] = ''                                                                        
    tokenizer = TweetTokenizer()
    detokenizer = MosesDetokenizer()

    for i, row in data2class.iterrows():
        # tokenise the tweet and apply pos tagging   
        tokens = tokenizer.tokenize(row['data'])
        tagged_tokens = add_pos_tags(tokens)

        # count tags per tweet
        nounCount, adjCount, advCount, verbCount = count_pos_tags(tagged_tokens)
        
        # add feature values
        data2class.at[i,'nounCount'] = nounCount
        data2class.at[i,'adjCount'] = adjCount
        data2class.at[i,'advCount'] = advCount
        data2class.at[i,'verbCount'] = verbCount

        # add lexicon-based SA score token
        lexScore = row['lexScore']
        tagged_tokens.append(f'lex_{lexScore}')

        # detokenise the tweet
        tweet = detokenizer.detokenize(tagged_tokens)
        data2class.at[i,'data'] = tweet

    return data2class


data_path = helper.data_path
results_path = helper.results_path + 'Iteration3\\'
resources_path = helper.resources_path
base_file = '_model_results.txt'

# import the best performing dataset
data = pd.read_csv(data_path + 'dataset-lex.csv').dropna()
data2class = data[data.target_names != 'neutral']

# split the data
X1 = data2class['data']
y = data2class['target']

# initialise classifiers
classifier_names = models.classifier_names
classifiers = models.classifiers
n_feats = models.n_feats

# define scoring metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# pre-process the data
preprocessed_data = preprocess(data2class)

# output preprocessed dataset for inspection
preprocessed_data.to_csv(f'{data_path}P2I3Preprocessed_data.csv', index=False)

# extract newly engineered features
new_feats = []
new_feats.append(preprocessed_data['nounCount'].astype(str).astype(int))
new_feats.append(preprocessed_data['adjCount'].astype(str).astype(int))
new_feats.append(preprocessed_data['advCount'].astype(str).astype(int))
new_feats.append(preprocessed_data['verbCount'].astype(str).astype(int))

# initialise vectorizers
tfidfs = {
    'UniGram': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english'),
    'BiGram': TfidfVectorizer(encoding='utf-8', ngram_range=(2, 2), stop_words='english'),
    'Combi': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 2), stop_words='english')
}

# perform cross-validation
for key,value in tfidfs.items():
    tfidf = value
    X_tfidf = tfidf.fit_transform(preprocessed_data)
    X_new = X_tfidf
    # add engineered features to TF-IDF csr matrix
    for new_feat in new_feats:
        X_new = combine_features(X_new, new_feat)
    
    results = {}
    basepath = f'{results_path}{key}\\'
    for i, clf in enumerate(classifiers):
        print(f'\nClassifier: {classifier_names[i]} \nStart Time: {dt.datetime.now()}\n')
        # initialise feature selector with k value needed
        selector = SelectKBest(score_func=chi2, k=n_feats[i])
        # intiialise the pipeline to be used for cross-validation
        pipe = make_pipeline(selector, clf)
        scores = helper.perform_cv(pipe, X_new, y, scoring)
        # output averaged results
        helper.write_avg_results(scores, classifier_names[i], f'{basepath}{key}{base_file}')
        results[classifier_names[i]] = pd.DataFrame.from_dict(scores)
        print(f'End Time: {dt.datetime.now()}')
    # output full cross-validation results
    helper.output_cv_results(results, basepath)
