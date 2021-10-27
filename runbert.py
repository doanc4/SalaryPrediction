import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW, BertConfig, DistilBertConfig
from sklearn.model_selection import train_test_split
import pickle


from nlp import get_bert_embeddings, SentenceDataset
from preprocessing import get_bin

df = pd.read_csv('DC_biotech_postings_May2020_to_July2021_subset.csv')
df = df.drop(['ID', 'PositionID', 'DateLastSeen', 'Date', 'Occupation2Code', 'Occupation2Name', 'Occupation6Code', 'Occupation6Name'], axis=1)
df = df.drop(['LocationCity', 'LocationMSA', 'LocationCounty', 'OccupationFunction', 'Occupation8Code'], axis=1)
#df = df.drop(['Certifications'], axis=1)

df['Certifications'] = df['Certifications'].fillna('None')



df = df.dropna()
df = df.reset_index(drop=True)

salary = list(df['SalaryAnnualEst'])
percentiles = list(np.percentile(salary, [0, 20, 40, 60, 80]))
text = df['ContentDescription']
df = df.drop(['Title', 'ContentDescription', 'SalaryAnnualEst'], axis=1)
df = df.reset_index(drop=True)




bin_labels = [get_bin(sal, percentiles) for sal in salary]

data = {
    'text': list(text),
    'label': bin_labels
}

data = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=.3, random_state=10)


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
dataset = SentenceDataset(train_encodings, list(y_train))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataloader = DataLoader(dataset, batch_size=4,shuffle=True)
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
config.num_labels = 5
model = DistilBertForSequenceClassification(config)
model.to(device)
model.train()

optim = AdamW(model.parameters(), lr=5e-5)
for epoch in range(3):
    print('Epoch: ', epoch)
    for i, batch in enumerate(dataloader):
        print(i)
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()


model.eval()

embeddings = [get_bert_embeddings(t, tokenizer, model, device) for text in text]

pickle.dump(embeddings, open('trained_embeddings_full.pkl', 'wb'))
