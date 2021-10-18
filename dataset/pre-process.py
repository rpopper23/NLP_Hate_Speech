import pandas as pd
import re
import nltk
from collections import OrderedDict
from nltk.corpus import stopwords
from collections import Counter
from urllib.parse import urlparse



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









##lunghezza tweet

#read file --> define the position of the file the position below IS ONLY AN EXAMPLE!!
df = pd.read_csv('dataset/final_dataset.csv', sep=',') 

df['tweet'] = df['tweet'].apply(str)

df['tweet'] = df['tweet'].apply(remove_emoji)

df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('www:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].apply(lambda x: re.split('html:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].str.replace(r'\S*twitter.com\S*', '')

df['tweet'] = df['tweet'].str.replace('[^a-zA-Z!?]', r' ')

df['tweet']  = df['tweet'].str.lower()
df["tweet"] = df["tweet"].apply(remove_stopwords)
df['tweet']= df['tweet'].str.replace(r'\s+', ' ')


df = df.drop_duplicates(subset=['tweet'])


df = df[df.tweet !='']

#tokenizing the tweet in vector
df['tweet'] = df['tweet'].apply(lambda x: tokenization(x.lower()))

#sort the databese based on ascendig date

#save dataset preprocessed
df.to_csv('dataset/final_dataset_post.csv',index=False, sep=',', encoding='utf-8', columns=['label', 'tweet'])
##lunghezza parole


df.to_csv("prova.csv", index=False)
print(df)
exit(0)
#remove links, urls, twitter.com and html


#remove everthing but characther

#remove words with only one characther and blank space

#everthing on lowercase
df['tweet']  = df['tweet'].str.lower()

#remove stopwords
df["tweet"] = df["tweet"].apply(remove_stopwords)
#lemmalization all tweet in tweets
df["tweet"] = df["tweet"].apply(lemmatize_words)
df['tweet']= df['tweet'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
#remove word repeated in a row
#df['tweet'] = (df['tweet'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))

#drop tweet that appears more then one times(keeping only the first)
df = df.drop_duplicates(subset=['tweet'])


df = df[df.tweet !='']

#tokenizing the tweet in vector
df['tweet'] = df['tweet'].apply(lambda x: tokenization(x.lower()))

#sort the databese based on ascendig date

#save dataset preprocessed
df.to_csv('dataset/final_dataset.csv',index=False, sep=',', encoding='utf-8', columns=['label', 'tweet'])