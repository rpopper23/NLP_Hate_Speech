import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def tf_idf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    return X

df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)
df = df[:2000]
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)


X = tf_idf(corpus)

num_training = int(len(df)*0.8)
X_train = X[:num_training]
X_test = X[num_training:]
y_train = df["label"][:num_training]
y_test = df["label"][num_training:]


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