from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk

def create_corpus():
    df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
    df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
    corpus = []
    for i in range(len(df)):
        text = df["tweet"].iloc[i]
        corpus.append(text)
    
    return corpus

def tf_idf():
    corpus = create_corpus()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    return X


def word2vec():
    corpus = create_corpus()
    model = Word2Vec(sentences=corpus, vector_size=100, window=2, min_count=1, workers=4)
    model.save("word2vec.model")
    return model


def doc2vec():
    corpus = create_corpus()
    #for i in range(len(corpus)):
    #    corpus[i] = (nltk.word_tokenize(corpus[i]))
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    model.save("doc2vec.model")
    return model

def n_grams(n=2):
    corpus = create_corpus()
    # from nltk.util import pad_sequence
    # for i in range(len(corpus)):
    #     corpus[i] = nltk.word_tokenize(corpus[i])
    #     corpus[i] = list(pad_sequence(corpus[i],
    #                pad_left=True, left_pad_symbol="<s>",
    #                pad_right=True, right_pad_symbol="</s>",
    #                n=2))
    #     corpus[i] = str(corpus[i])s
    #     corpus[i] = corpus[i].replace('[^\w\s]','')
    #     exit(0)
                
    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(n, n))
    X2 = vectorizer2.fit_transform(corpus)
    print(X2.toarray())
    return X2


print(n_grams(3))
#print(tf_idf())
#print(word2vec())
#print(doc2vec())



