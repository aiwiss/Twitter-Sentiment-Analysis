import got3
import csv
import os
import helper

data_path = helper.data_path

# Declare the data collection period
sinceDate = '2019-04-11'
untilDate = '2019-04-13'

tweets = []

# Get 1000 tweets for the period desired and append to tweets list
tweetCriteria = got3.manager.TweetCriteria().setQuerySearch('brexit').setSince(sinceDate).setUntil(untilDate).setMaxTweets(1000)
tweets += got3.manager.TweetManager.getTweets(tweetCriteria)

# Open the csv file and output tweets received
with open(data_path + 'BrexitSampleeee.csv', 'w', encoding='utf-8') as csv_output:
    tweetWriter = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tweetWriter.writerow(['ID', 'Date', 'Tweet'])
    for tweet in tweets:
        tweetWriter.writerow([tweet.id, tweet.date, tweet.text])