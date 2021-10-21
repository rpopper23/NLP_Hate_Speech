import pandas as pd


def split():
    df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
    df_shuffled=df.sample(frac=1).reset_index(drop=True)
    n_train = int(len(df)*0.80)
    train = df_shuffled[:n_train]
    test = df_shuffled[n_train:]


    train.to_csv("dataset/train_hate_speech.csv", index=False)
    test.to_csv("dataset/test_hate_speech.csv", index=False)