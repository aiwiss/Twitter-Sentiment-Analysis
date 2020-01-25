import datetime as dt
import got3
import csv
import helper

data_path = helper.data_path

# Define the period for data collection
sinceDate = '2019-04-11'
untilDate = '2019-04-12'
print("Downloading start time: ", dt.datetime.now())

# Open file stream to output the data
with open(data_path + 'BrexitDatasetRaw.csv', 'w', encoding='utf-8') as csv_output:
    
    # Initialise the csv writer to output the data in correct csv format
	tweetWriter = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	tweetWriter.writerow(['ID', 'Date', 'Tweet'])

	# Set tweets download criteria
	tweetCriteria = got3.manager.TweetCriteria().setLang('en').setQuerySearch('brexit OR referendum OR theresa OR farage OR remain OR leave').setSince(sinceDate).setUntil(untilDate).setMaxTweets(500)
	tweets = got3.manager.TweetManager.getTweets(tweetCriteria)

	# Output received tweets to csv file
	for tweet in tweets:
		tweetWriter.writerow([tweet.id, tweet.date, tweet.text])

print("Downloading end time: ", dt.datetime.now())
print("Tweets received: ", len(tweets))
