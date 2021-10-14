import pandas as pd

df = pd.read_csv("/Users/elia/Desktop/data_hate_speech/final_pre_processing.csv", sep=",")

df_shuffled=df.sample(frac=1).reset_index(drop=True)

#print(df_shuffled)
n_train = int(len(df)*0.8)
n_test = len(df) - n_train
train = df_shuffled[:n_train]
test = df_shuffled[n_train:]

train.to_csv("train_hate_speech.csv", index=False)
test.to_csv("test_hate_speech.csv", index=False)