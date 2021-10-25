from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

tweet_tokenizer = TweetTokenizer()


def doc2vec(corpus):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
    #model.save("doc2vec.model")
    return model

df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)


df = df[:30000]

df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)


parsed_tweet = []

for info in corpus:
    l = " ".join(tweet_tokenizer.tokenize(info.lower())).split(" ")   
    parsed_tweet.append(l)


num_training = int(len(df)*0.8)

model = doc2vec(parsed_tweet)

document_vectors = [model.infer_vector(s) for s in tqdm(parsed_tweet)]



X_train, X_test, y_train, y_test = train_test_split(document_vectors, df.label,test_size=0.8, random_state=42)


print(".....")
clf = SVC(probability=False, kernel='rbf')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
