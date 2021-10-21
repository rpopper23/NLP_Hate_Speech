import os
import re
import pandas as pd
import numpy as np
import glob
### student code here: import the needed modules from sci-kit learn ###
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import text 
from dataset.split_test_train import split



for i in range(10):
    split()
    data = pd.read_csv("dataset/train_hate_speech.csv", sep=",")

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    X = vectorizer.fit_transform(data.tweet)

    y = data["label"].to_list()


    clf = MultinomialNB()
    clf.fit(X, y)

    test = pd.read_csv("dataset/test_hate_speech.csv", sep=",")


    Z = vectorizer.transform(test.tweet)
    predicted = clf.predict(Z)

    y_test = test.label

    print(accuracy_score(y_test, predicted))