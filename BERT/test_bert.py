from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split 
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from sklearn.metrics import accuracy_score
class BERT_Arch(nn.Module):
    
    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask,return_dict=False)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x

df = pd.read_csv("dataset/final_dataset_post.csv", sep=",")
df=df.sample(frac=1).reset_index(drop=True)
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')

df = df[:500]
corpus = []
for i in range(len(df)):
    text = df["tweet"].iloc[i]
    corpus.append(text)

bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',return_dict=False)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    corpus,
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)



test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(df.label.tolist())



from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


device = torch.device("cpu")

print(device)

path = 'saved_weights_5.pt'
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

model = torch.load(path,map_location=torch.device('cpu'))


with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)

test_y = test_y.numpy()
print(accuracy_score(test_y, preds))
print(classification_report(test_y, preds))

