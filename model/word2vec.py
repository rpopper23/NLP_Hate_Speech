import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

tweet_tokenizer = TweetTokenizer()

def word2vec(corpus):
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    #model.save("word2vec.model")
    return model


df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)
df = df[:3000]
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)

parsed_tweet = []

for info in corpus:
    l = " ".join(tweet_tokenizer.tokenize(info.lower())).split(" ")

    
    parsed_tweet.append(l)

model = word2vec(parsed_tweet)

word2vec_feature = []
for tweet in parsed_tweet:
    average_vec = np.zeros(100)
    for word in tweet:
        if word in model.wv:
            average_vec += (model.wv[word] / len(tweet))
        else:
            pass
    word2vec_feature.append(average_vec)


num_training = int(len(df)*0.8)
X_train, X_test, y_train, y_test = train_test_split(word2vec_feature, df.label,test_size=0.8, random_state=42)



print(".....")
clf = SVC(probability=False, kernel='rbf')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


clf = LogisticRegression(random_state=42).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))