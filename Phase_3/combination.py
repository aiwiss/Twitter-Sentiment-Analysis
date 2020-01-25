import os, datetime as dt, re
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import models, helper
from nltk.sentiment.util import mark_negation
from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer
from nltk import pos_tag
from nltk.corpus import wordnet

data_path = helper.data_path
results_path = helper.results_path + 'Combination\\'
resources_path = helper.resources_path
base_file = '_model_results.txt'

class SlangHandler:
    # helper function to read slang words
    def load_slang(self):
        with open(resources_path + 'slang_dictionary.txt', encoding='utf-8') as f:
            text = f.read().splitlines()
            result = {}
            for line in text:
                l = line.split(':')
                result[l[0]] = l[1]
            return result

    # helper function to read abbreviations dictionary
    def load_abbreviations(self):
        with open(resources_path + 'abbreviations_dictionary.txt', encoding='utf-8') as f:
            text = f.read().splitlines()
            result = {}
            for line in text:
                l = line.split('@')
                result[l[0]] = l[1]
            return result
    
    # pre-process tweets by replacing slang and abbreviations
    def preprocess_tweet(self, tweet, slang_keys, slang_dictionary, abbr_keys, abbr_dictionary):
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

class PosTagHandler:
    # helper function to add generated pos-tag features into the csr matrix
    # produced by TF-IDF vectorizer
    def combine_features(self, X1, feature):
        X2 = feature.reset_index()
        X2 = X2.iloc[:,[1]].values
        combined = sp.hstack((X1, X2), format='csr')
        return combined

    # helper function to assign pos-tags for words
    def add_pos_tags(self, tokens):
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
    def count_pos_tags(self, tokens):
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

    def preprocess_data(self, X):
        # add columns to DataFrame for features
        X['nounCount'] = ''
        X['adjCount'] = ''
        X['advCount'] = ''
        X['verbCount'] = ''                                                                        
        tokenizer = TweetTokenizer()
        detokenizer = MosesDetokenizer()

        for i, row in X.iterrows():
            # tokenise the tweet and apply pos tagging   
            tokens = tokenizer.tokenize(row['data'])
            tagged_tokens = self.add_pos_tags(tokens)

            # count tags per tweet
            nounCount, adjCount, advCount, verbCount = self.count_pos_tags(tagged_tokens)
            
            # add feature values
            X.at[i,'nounCount'] = nounCount
            X.at[i,'adjCount'] = adjCount
            X.at[i,'advCount'] = advCount
            X.at[i,'verbCount'] = verbCount

            # detokenise the tweet
            tweet = detokenizer.detokenize(tagged_tokens)
            X.at[i,'data'] = tweet

        return X


# starting with URL-handled tweets
X = pd.read_csv(f'{data_path}P2I6Preprocessed_tweets.csv', sep='"""')
data = pd.read_csv(f'{data_path}dataset-lex.csv')
y = data[data.target_names != 'neutral']['target']

# initialise classifiers
classifier_names = models.classifier_names
classifiers = models.classifiers
n_feats = models.n_feats

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

#import slang and abbreviation dictionaries
slang_handler = SlangHandler()
slang = slang_handler.load_slang()
abbreviations = slang_handler.load_abbreviations()

# slang and abbreviation handling
slang_keys = list(slang.keys())
abbreviations_keys = list(abbreviations.keys())
for i, row in X.iterrows():
    tweet = row['data']
    new_tweet = slang_handler.preprocess_tweet(tweet, slang_keys, slang, abbreviations_keys, abbreviations)
    X.at[i,'data'] = new_tweet


# pos-tag handling
pos_tag_handler = PosTagHandler()
X = pos_tag_handler.preprocess_data(X)

# hashtag handling
for i, row in X.iterrows():
    if '#' in row['data']:
        text = row['data']
        text = re.sub('#', '', text)
        X.at[i,'data'] = text

# negation handling
tokenizer = TweetTokenizer()
detokenizer = MosesDetokenizer()
for i,row in X.iterrows():
    tokens = tokenizer.tokenize(row['data'])
    marked_tokens = mark_negation(tokens)
    text = detokenizer.detokenize(marked_tokens)
    X.at[i,'data'] = text

# extract pos-tag based engineered features
pos_feats = []
pos_feats.append(X['nounCount'].astype(str).astype(int))
pos_feats.append(X['adjCount'].astype(str).astype(int))
pos_feats.append(X['advCount'].astype(str).astype(int))
pos_feats.append(X['verbCount'].astype(str).astype(int))

# initialise vectorizers
tfidfs = {
    'UniGram': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1), stop_words='english'),
    'BiGram': TfidfVectorizer(encoding='utf-8', ngram_range=(2, 2), stop_words='english'),
    'Combi': TfidfVectorizer(encoding='utf-8', ngram_range=(1, 2), stop_words='english')
}

# perform cross-validation
for key,value in tfidfs.items():
    tfidf = value
    X_tfidf = tfidf.fit_transform(X['data'])
    X_new = X_tfidf
    # add engineered features to TF-IDF csr matrix
    for feat in pos_feats:
        X_new = pos_tag_handler.combine_features(X_new, feat)
    
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