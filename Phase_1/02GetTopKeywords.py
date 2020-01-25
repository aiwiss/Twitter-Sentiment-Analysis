from collections import Counter
from string import punctuation
import csv
import nltk 
import helper

data_path = helper.data_path

def get_top_keywords(tweets):
    # Exclude stop words from counting
    stopwordsList = set(nltk.corpus.stopwords.words('english'))
    # Initialise word counter
    keywords = Counter()
    for tweet in tweets:
        # Tokenise the tweet
        tokens = tweet['Tweet'].split()
        # Update the word counts if current word is not a stopword
        keywords.update(t.lower().rstrip(punctuation) for t in tokens if t not in stopwordsList)
    return [k for k in keywords.most_common(50)]

tweets = []

with open(data_path +'BrexitSample.csv', encoding='utf-8') as csv_data:
    # Import csv data as a dictionary
    reader = csv.DictReader(csv_data)
    for line in reader:
        # Check if line is not empty then append to list
        if any(x.strip() for x in line):
            tweets.append(line)

# Extract top keywords
topKeywords = get_top_keywords(tweets)
with open(data_path + 'top50keywords.txt', 'w', encoding='utf-8') as f:
    # Handle printing the tuple
    f.write('\n'.join('%s %s' % k for k in topKeywords))
        