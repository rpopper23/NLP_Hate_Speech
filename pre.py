import pandas as pd
import re
import nltk
from collections import OrderedDict
from nltk.corpus import stopwords
from collections import Counter
from urllib.parse import urlparse
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



STOPWORDS = set(stopwords.words('english'))


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


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])







##lunghezza tweet

#read file --> define the position of the file the position below IS ONLY AN EXAMPLE!!
df = pd.read_csv('#food.csv', sep=',') 

df['tweet'] = df['tweet'].apply(str)


df['tweet']  = df['tweet'].str.lower()

df = df.drop_duplicates(subset=['tweet'])

df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('www:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('html:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].str.replace(r'\S*twitter.com\S*', '')





df["tweet"] = df["tweet"].apply(remove_stopwords)
df['tweet'] = df['tweet'].apply(remove_emoji)


df['tweet'] = df['tweet'].str.replace('[^a-zA-Z0-9]', r' ')

avg_word_len = []
for i in range(len(df)):
    t = nltk.word_tokenize(df["tweet"].iloc[i])
    word_len = []
    for item in t:
        word_len.append(len(item))
    avg_word_len.append(np.mean(word_len))
print(avg_word_len)
df["avg_word_len"] = avg_word_len



df['tweet']= df['tweet'].str.replace(r'\s+', ' ')


df = df[df.tweet !='']


tweet = []
for i in range(len(df)):
    sentence = df["tweet"].iloc[i]
    tweet.append([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])

df["tweet"] = tweet


df.to_csv('food.csv',index=False, sep=',', encoding='utf-8', columns=['label', 'tweet'])

