from sentence_transformers import SentenceTransformer

import pandas as pd



df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)

df = df[:10000]
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')


corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = model.encode(corpus)

