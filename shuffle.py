import pandas as pd

df = pd.read_csv("dataset/final_dataset_post.csv",sep=",")

df=df.sample(frac=1).reset_index(drop=True)
df1 = df[:50000]

df1.to_csv("train.csv",sep=",",index=False)

df1 = df[50000:]

df1.to_csv("test.csv",sep=",",index=False)
