import tweepy
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import preprocessor
import emoji
from textblob import TextBlob
from textblob.exceptions import NotTranslated
from pymongo import MongoClient
import time

import pprint # remove

import os, sys
from dotenv import load_dotenv

BASEDIR = os.path.join(os.getcwd(), "../../")
sys.path.append(BASEDIR)

#Load environment Variables
load_dotenv(os.path.join(BASEDIR, '.env'))

# Environment Variable
consumer_key=os.getenv("CONSUMER_KEY")
consumer_secret=os.getenv("CONSUMER_SECRET")
access_token=os.getenv("ACCESS_TOKEN")
access_token_secret=os.getenv("ACCESS_TOKEN_SECRET")
mongodb_url = os.getenv("MONGODB_URL")
num_tweets = 50

# Connect and save data to mongodb
client = MongoClient(mongodb_url)
db=client['Tweet_Sentiment']
coll = db["sentiment_data"]

# Get all the current documents from mongodb
past_tweets = []
for doc in coll.find():
    past_tweets.append(doc['tweet'])

# Check if the tweet already exists
def already_exists(tweet):
    if tweet in past_tweets:
        return True
    else:
        past_tweets.append(tweet)
        return False


saved_data = []
# Saves data to saved_data
def save_data(clean_tweet_en, date, polarity, subjectivity):
    data = {
            "tweet": clean_tweet_en,
            "date": date,
            "polarity": polarity,
            "subjectivity": subjectivity
            }
    
    try:
        if already_exists(clean_tweet_en):
            print(f'[INFO] Already Exists {clean_tweet_en}')
        else:
            saved_data.append(data)
            print("[INFO] Tweet Added")
            
    except Exception as e:
        print("Error on saving data")
        print(e)
        
# Pushes data as a batch to mongodb
def save_mongo_data():
    try:
        coll.insert_many(saved_data)
    except Exception as e:
        print("Error on pushing data to mongodb")
        print(e)
    
#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def mongo_conn(self):
        self.mongo_db = mongo()
        
    
    def on_status(self, status):
        # Preprocess Tweet
        clean_tweet = preprocess_data(str(status.text))

        # This Functionality is removed because there are a limited number of requests that can be made to this API
        # Translate tweet to english
        # try:
        #     clean_tweet_en = str(TextBlob(clean_tweet).translate(to='en'))
        # except NotTranslated:
        clean_tweet_en = clean_tweet
            
        # Compute Sentiment of the tweet
        polarity, subjectivity = tweet_sentiment(clean_tweet_en)
        
        '''
        - If the tweet is not already in the databse
        - Add the date, tweet, polarity, subjectivity of that tweet to the databse
        '''
        save_data(clean_tweet, date=status.created_at, polarity=polarity, subjectivity=subjectivity)
        
        # Scan a determined number of tweets
        if len(saved_data) < num_tweets:
            return True
        else:
            return False
        
class twitter_sentiment:
    def __init__(self, filter_string):
        self.filter = filter_string
        self.api = self.conn()
        
    def conn(self):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)
        
        return api

    def stream(self):
        myStreamListener = MyStreamListener()
        myStream = tweepy.Stream(auth = self.api.auth, listener=myStreamListener)
        myStream.filter(track=self.filter) # filter stream
        save_mongo_data() # Saving the new data to mongodb

# Preprocess the text data
def preprocess_data(tweet):
    # Replace emoji with their written meaning'
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(':', ' ')
    # Replace #
    tweet = tweet.replace('#', '')
    # Replace @
    tweet = tweet .replace('@', '')
    # Replace _ with ' '
    tweet = tweet.replace('_', ' ')
    
    # tweet preprocessing library
    tweet = preprocessor.clean(tweet)
    
    # Double check that the tweet is clean
    #HappyEmoticons
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
        ])
    
    # Sad Emoticons
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
        ])
    
    #Emoji patterns
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
    
    #combine sad and happy emoticons
    emoticons = emoticons_happy.union(emoticons_sad)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # tokenize tweet, i.e array of words, remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(tweet)
    
    #after tweepy preprocessing the colon symbol left remain after 
    tweet = re.sub(r':', '', tweet)
    #removing mentions  
    tweet = re.sub(r'‚Ä¶', '', tweet)
    
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    
    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    
    # Update stopwords
    stop_words.add('https')
    
    filtered_tweet = []
    
    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons:
            filtered_tweet.append(w.lower())
            
    return(' '.join(filtered_tweet))    


# Compute the sentiment of the tweet
# Return the polarity and subjectivity of the tweet
def tweet_sentiment(tweet):
    sentiments = TextBlob(tweet)
    return sentiments.sentiment.polarity, sentiments.sentiment.subjectivity
        
if __name__ == '__main__':
    filter_strings = ["Logan", "Paul", "ksi"]
    sentiment = twitter_sentiment(filter_strings)
    sentiment.stream()
    
    
    
'''
TODO :
- At the start of the program create a json file of all the data from mongodb
- On every iteration of the stream check to see if the data is present inside the json file
- if not present, add it to the data and another empty array which will later be pushed on to mongo
- This new array will contain all the new json data that already doesn't exist on mongo
- This will reduce connection to mongodb from infinite to 2 per stream iteration

How to end the stream on command?
- is there a number n for which the program only runs through n items from the stream 
'''