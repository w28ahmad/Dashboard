import tweepy

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

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)

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
        myStream.filter(track=self.filter)

if __name__ == '__main__':
    filter_strings = ["Amazon", "AMZN", "stock"]
    sentiment = twitter_sentiment(filter_strings)
    sentiment.stream()