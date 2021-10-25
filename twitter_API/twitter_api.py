import os
import tweepy as tw
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

consumer_key= os.getenv("consumer_key")
consumer_secret= os.getenv("consumer_secret")
access_token= os.getenv('access_token')
access_token_secret= os.getenv('access_token_secret')


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)



def get_username(name):
    user_objects = api.search_users(q = name, page="1",count=10)
    return user_objects


def get_info_user(name):
    user_objects = api.get_user(screen_name = name)
    return user_objects

def get_tweets(name,number):    
    tweet_objects = api.user_timeline(screen_name = name, count = number)
    return  tweet_objects
def get_tweets_no_re(name,number):    
    tweet_objects = api.user_timeline(screen_name = name, count = number, exclude_replies=True)
    return  tweet_objects

def get_tweets_once(name):    
    tweet_objects = api.user_timeline(screen_name = name, count = 100) #exclude_replies=True
    return  tweet_objects
def all_tweets(name):    
    tweet_objects = api.user_timeline(screen_name = name, count=5000)
    return tweet_objects
def all_tweets_no_re(name):    
    tweet_objects = api.user_timeline(screen_name = name, count=2000, exclude_replies=True)
    return tweet_objects
def get_stat(id):
    tweets_objects = api.lookup_statuses(id = list(id))
    return tweets_objects
#-----------------------------------------------------------------------------------------------------------------
def get_info_user(name):
    user_objects = api.get_user(user_id = name)
    return user_objects

def user(users):
    user_list = get_username(users)
    user_name = [user.name for user in user_list]  
    user_screen = [user.screen_name for user in user_list]
    description = [user.description for user in user_list]
    verified = [user.verified for user in user_list]
    followers_count  = [user.followers_count for user in user_list]  
    profile_image = [user.profile_image_url for user in user_list]  
    
    df = pd.DataFrame()
    df['user_name'] = user_name
    df['user_screen'] = user_screen
    df['description'] = description
    df['verified'] = verified
    df['followers_count'] = followers_count
    df['profile_image'] = profile_image
    
    return df

def user_id(id):
    user_list = get_info_user(id)
    df = {}
    df['user_name'] = user_list.name
    df['user_screen'] = user_list.screen_name
    df["id_user"] = user_list.id
    df['description'] = user_list.description
    df['verified'] = user_list.verified
    df['followers_count'] = user_list.followers_count
    
    return df
def username(username_screen_name,number):
    tweet_objects = get_tweets(username_screen_name,number)
    text = [tweet.text for tweet in tweet_objects]
    dates = [tweet.created_at for tweet in tweet_objects]
    retweet_count = [tweet.retweet_count for tweet in tweet_objects]
    favorite_count = [tweet.favorite_count for tweet in tweet_objects]
    id = [tweet.id_str for tweet in tweet_objects]
    df = pd.DataFrame()
    df['date'] = dates
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")    
    df['text'] = text
    df['retweet_count'] = retweet_count
    df['like'] = favorite_count  
    df['date'] = df['date'].apply(str)
    df["id"] = id
    return df

def get_status(id):
    print(id)
    tweet_objects = get_stat(id)
    for item in tweet_objects:
        print("ciao")
    return tweet_objects

def search_user_retweet(q):
    tweet_object = api.get_retweeter_ids(id = q, count=10)
    return tweet_object

def search_tweet(q):
    tweet_objects = api.search_tweets(q = q, result_type="mixed", count=100)
    
    text = [tweet.text for tweet in tweet_objects]
    dates = [tweet.created_at for tweet in tweet_objects]
    retweet_count = [tweet.retweet_count for tweet in tweet_objects]
    favorite_count = [tweet.favorite_count for tweet in tweet_objects]
    id = [tweet.id_str for tweet in tweet_objects]
    df = pd.DataFrame()
    df['date'] = dates
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")    
    df['text'] = text
    df['retweet_count'] = retweet_count
    df['like'] = favorite_count  
    df['date'] = df['date'].apply(str)
    df["id"] = id
    
    return df

print("ok")

hast = ["#food","#money",'#gun']

import timeit
start = timeit.default_timer()
diff = 0

while (diff < 1000000):
    for item in hast:
        tweet = []
        ttt = search_tweet(item)
        print("ttt")
        for twit in ttt.text:
            twit = twit.replace("\n","")
            tweet.append(twit)
        
        print(len(tweet))
        name = item + '.txt'
        with open(name, 'a') as filehandle:
            for listitem in tweet:
                listitem = listitem.replace(",","")
                if 'RT' not in listitem:
                    if item in listitem:
                        filehandle.write('%d,%s \n' % (0,listitem))  
        
    stop = timeit.default_timer()
    diff = stop - start
    import time
    print(diff)
    
    time.sleep(10)
        

