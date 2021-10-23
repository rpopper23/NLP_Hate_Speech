

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split 

import pandas as pd



df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)

df = df[:100]
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')


corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = sbert_model.encode(corpus)

X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, df.label,test_size=0.8, random_state=42)

from sklearn.svm import SVC

print(".....")
clf = SVC(probability=False, kernel='rbf')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
