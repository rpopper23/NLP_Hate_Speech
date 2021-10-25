import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)


df = df[:3000]

df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)



sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = sbert_model.encode(corpus)

X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, df.label,test_size=0.8, random_state=42)


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
