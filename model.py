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
tweet_tokenizer = TweetTokenizer()

def tf_idf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    return X


def word2vec(corpus):
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return model


def doc2vec(corpus):
    #for i in range(len(corpus)):
    #    corpus[i] = (nltk.word_tokenize(corpus[i]))
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
    model.save("doc2vec.model")
    return model

def n_grams(corpus):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
    X2 = bigram_vectorizer.fit_transform(corpus)

    return X2


#print(n_grams(corpus,3))

df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)

df = df[:10000]

df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)

#X = tf_idf(corpus)
#X = n_grams(corpus)


parsed_tweet = []

for info in corpus:
    l = " ".join(tweet_tokenizer.tokenize(info.lower())).split(" ")
    filtered_sentence = [w for w in l if not w in string.punctuation
                         and ( w[0] != '@' and w[0] != '#' and w[:4] != 'http' )]
    
    parsed_tweet.append(filtered_sentence)

model = doc2vec(parsed_tweet)

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



'''
For td-idf and n-grams
# X_train = X[:num_training]
# X_test = X[num_training:]
# y_train = df["label"][:num_training]
# y_test = df["label"][num_training:]
'''




X_train =word2vec_feature[:num_training]
X_test =word2vec_feature[num_training:]
y_train = df["label"][:num_training]
y_test = df["label"][num_training:]
from sklearn.svm import SVC

print(".....")
clf = SVC(probability=False, kernel='rbf')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
# predict and evaluate predictions

#print(word2vec())
#print(doc2vec())



