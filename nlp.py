import string
import re
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np


stop_words = set(stopwords.words('english'))
stop_words.remove('i')
def clean_text(text, lemmatizer=None, stemmer=None, return_tokenized=False):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    text = re.sub('[^a-zA-Z0-9_\s]', '', text)
    text_tokens = word_tokenize(text)
    text_tokens = [tok for tok in text_tokens if tok not in stop_words]
    if lemmatizer:
        text_tokens = [lemmatizer.lemmatize(tok) for tok in text_tokens]
    if stemmer:
        text_tokens = [stemmer.stem(tok) for tok in text_tokens]
    if return_tokenized:
        return text_tokens
    text = ' '.join(text_tokens)
    return text


def get_bert_embeddings(text, tokenizer, model, device):
    """Gets the embeddings of the last layer's CLS token for each input sequence"""
    encodings = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    hidden_state = model(input_ids, attention_mask, output_hidden_states=True).hidden_states[-1].detach().numpy().mean(axis=1)
    return hidden_state


class SentenceDataset(Dataset):
    """Custom Dataset class for fine-tuning BERT"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_embs(text, model, lemmatizer):
    """Gets averaged word2vec embeddings for a given piece of text"""
    text_tokens = clean_text(text, lemmatizer=lemmatizer, return_tokenized=True)
    embs = []
    for tok in text_tokens:
        if tok in model.wv.key_to_index:
            tok_emb = model.wv[tok]
            embs.append(tok_emb)
    return np.mean(np.array(embs), axis=0)