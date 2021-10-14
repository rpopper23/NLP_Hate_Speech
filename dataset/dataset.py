import pandas as pd

df1 = pd.read_csv("/Users/elia/Desktop/data_hate_speech/2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv", sep=",")

tweet1 = df1["text"].to_list()
label1 = df1["label"].to_list()

for i in range(len(label1)):
    if label1[i] == 'hate':
        label1[i] = 1
    elif label1[i] == 'nothate':
        label1[i]=0

df2 = pd.read_csv("/Users/elia/Desktop/data_hate_speech/labeled_data.csv", sep=",")
tweet2 = df2["tweet"].to_list()
label2 = df2["class"].to_list()

for i in range(len(label2)):
    if label2[i] == 0:
        label2[i] =1 
    elif label2[i]==1:
        label2[i] =1
    elif label2[i] == 2:
        label2[i] =0

df3 = pd.read_csv("/Users/elia/Desktop/data_hate_speech/train.csv", sep=",")
tweet3 = df3["tweet"].to_list()
label3 = df3["label"].to_list()


overall_tweet = [*tweet1,*tweet2,*tweet3]
overall_label = [*label1,*label2,*label3]

df = pd.DataFrame()
df["label"] = overall_label
df["tweet"] = overall_tweet

df.to_csv("final_dataset.csv",index=False)

h = 0
nh = 0
for item in overall_label:
    if item == 1:
        h += 1
    else:
        nh+=1
o = h + nh
print(h/o * 100,nh/o*100)

