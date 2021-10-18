import pandas as pd
import numpy as np
import regex as re

def tokenization(text):
    text = re.split('\W+', text)
    return text

df = pd.read_csv("dataset/final_pre_processing.csv",sep=",")
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')

#tokenize each tweet
df['tweet'] = df['tweet'].apply(lambda x: tokenization(x.lower()))

tweets = df.tweet.to_list()

lenght = []

#to save
word_lenght = []

for item in tweets:
    lenght.append(len(item))
    l = []
    for word in item:
        l.append(len(item))
    
    word_lenght.append(np.mean(l))



print("avarage token in a tweet {0}".format(np.mean(lenght)))


print("avarage lenght in  tweets {0}".format(word_lenght))
