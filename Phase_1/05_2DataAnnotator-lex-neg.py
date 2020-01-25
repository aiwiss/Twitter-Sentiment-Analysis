# Lexicon-based data annotator with handled negation

import csv
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk import pos_tag
import string
import datetime as dt
import helper

data_path = helper.data_path
resources_path = helper.resources_path
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
negation_words = []
negation_prefixes = []

# read the list of negationm words and negation prefixes
with open(resources_path + 'negation_words.txt', encoding='utf-8') as f:
    negation_words = f.read().splitlines()
with open(resources_path + 'negation_prefixes.txt', encoding='utf-8') as f:
    negation_prefixes = f.read().splitlines()

# data investigation showed some defficiencies regarding negations, so additional checks in
# this function help to detect negation in those special cases as well
def is_negation(token):
    if token in negation_words:
        return True
    elif token == 't' or token.endswith(("n’t")):
        return True
    elif token.startswith(tuple(negation_prefixes)):
        return True
    else:
        return False

# read pre-processed data
with open(data_path + 'PreprocessedBrexitDataset-lex-neg.csv', encoding='utf-8') as csv_input, open(data_path + 'dataset-lex-neg.csv', 'w', encoding='utf-8', newline='') as csv_output:
    reader = csv.DictReader(csv_input)
    writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['date', 'data', 'target', 'target_names', 'lexScore'])
    reverse = False
    counter = 0
    print("Annotation start time: ", dt.datetime.now())
    for tweet in reader:
        sentiment_score = 0
        # assign the pos tag which is needed for SentiWordNet lexicon
        tagged_tweet = pos_tag(tokenizer.tokenize(tweet['data']))
        
        for token, penn_tag in tagged_tweet:
            # detect negation and punctuation
            if is_negation(token):
                reverse = True
            elif token.endswith(tuple(punctuations)):
                reverse = False
            # convert PENN pos tag to wordnet tag
            tag = helper.penn_to_wordnet_tag(penn_tag)
            if not tag:
                continue
            # lemmatize the word
            lemma = lemmatizer.lemmatize(token, pos=tag)
            if not lemma:
                continue
            # get the synonyms for word as recommended by the methodology
            synsets = wordnet.synsets(lemma, pos=tag)
            if not synsets:
                continue
            # take the first, most popular synonym
            first_synset = synsets[0]
            sentiment_synset = sentiwordnet.senti_synset(first_synset.name())
            synset_score = sentiment_synset.pos_score() - sentiment_synset.neg_score()

            # handle negation
            if reverse:
                synset_score = synset_score * -1
            
            sentiment_score += synset_score
        # assign the polarity label based on sentiment score according to the threshold set
        if sentiment_score > 0.4:
            writer.writerow([tweet['date'], tweet['data'], '1', 'positive', sentiment_score])
        elif sentiment_score < -0.4:
            writer.writerow([tweet['date'], tweet['data'], '0', 'negative', sentiment_score])
        else:
            writer.writerow([tweet['date'], tweet['data'], '2', 'neutral', sentiment_score])

print("Annotation end time: ", dt.datetime.now())