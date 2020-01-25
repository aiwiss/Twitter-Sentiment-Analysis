# Lexicon-based data annotator

import csv
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk import pos_tag
from itertools import zip_longest
import datetime as dt
import helper

data_path = helper.data_path
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()

# read pre-processed data
with open(data_path + 'PreprocessedBrexitDataset-lex.csv', encoding='utf-8') as csv_input, open(data_path + 'dataset-lex.csv', 'w', encoding='utf-8', newline='') as csv_output:
    reader = csv.DictReader(csv_input)
    writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['date', 'data', 'target', 'target_names', 'lexScore'])
    counter = 0
    print("Annotation start time: ", dt.datetime.now())
    for tweet in reader:
        sentiment_score = 0
        # assign the pos tag which is needed for SentiWordNet lexicon
        tagged_tweet = pos_tag(tokenizer.tokenize(tweet['data']))
        
        for token, penn_tag in tagged_tweet:
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
            # calculare the score
            sentiment_score += sentiment_synset.pos_score() - sentiment_synset.neg_score()

        # assign the polarity label based on sentiment score according to the threshold set
        if sentiment_score > 0.4:
            writer.writerow([tweet['date'], tweet['data'], '1', 'positive', sentiment_score])
        elif sentiment_score < -0.4:
            writer.writerow([tweet['date'], tweet['data'], '0', 'negative', sentiment_score])
        else:
            writer.writerow([tweet['date'], tweet['data'], '2', 'neutral', sentiment_score])
        
print("Annotation end time: ", dt.datetime.now())