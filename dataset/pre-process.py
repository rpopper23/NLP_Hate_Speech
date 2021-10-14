import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from collections import OrderedDict
from nltk.corpus import stopwords
from collections import Counter
from urllib.parse import urlparse
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def tokenization(text):
    text = re.split('\W+', text)
    return text




#read file --> define the position of the file the position below IS ONLY AN EXAMPLE!!
df = pd.read_csv('/Users/elia/Desktop/data_hate_speech/listfile.csv', sep=',') 

#delete hours in date and cast to elements in columns to Date
#cast the tweet's text to string
df['tweet'] = df['tweet'].apply(str)

#remove emoji
df['tweet'] = df['tweet'].apply(remove_emoji)

#remove links, urls, twitter.com and html
df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('www:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('html:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].str.replace(r'\S*twitter.com\S*', '')

#remove everthing but characther
df['tweet'] = df['tweet'].str.replace('[^a-zA-Z]', ' ')

#remove words with only one characther and blank space
df['tweet']= df['tweet'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')

#everthing on lowercase
df['tweet']  = df['tweet'].str.lower()

#remove stopwords
df["tweet"] = df["tweet"].apply(remove_stopwords)
#lemmalization all tweet in tweets
df["tweet"] = df["tweet"].apply(lemmatize_words)
df['tweet']= df['tweet'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
#remove word repeated in a row
df['tweet'] = (df['tweet'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))

#drop tweet that appears more then one times(keeping only the first)
df = df.drop_duplicates(subset=['tweet'])


df = df[df.tweet !='']

#tokenizing the tweet in vector
df['tweet'] = df['tweet'].apply(lambda x: tokenization(x.lower()))

#sort the databese based on ascendig date

#save dataset preprocessed
df.to_csv('final_pre_processing.csv',index=False, sep=',', encoding='utf-8', columns=['label', 'tweet'])