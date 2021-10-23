from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import numpy as np
from nltk.tokenize import TweetTokenizer
import string
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn import preprocessing
df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)

df = df[:10000]



X=df['tweet'].to_list()
y=df['label'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=10)
Vectorizer = TfidfVectorizer(max_df=0.9,ngram_range=(1, 2))
TfIdf=Vectorizer.fit(X_train)
X_train=TfIdf.transform(X_train)

model = SVC(C=0.1)
model.fit(X_train,y_train)

le = preprocessing.LabelEncoder()

X_test=TfIdf.transform(X_test)
y_pred=model.predict(X_test)
y_test=le.transform(y_test)
print(classification_report(y_test, y_pred))