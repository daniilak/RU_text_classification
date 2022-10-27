# -*- coding: utf-8 -*-

DADASETID = int(input("Выберите датасет:\n1 - Студенты\n2 - Профессионалы\nВведите число:"))


DADASET_NAME = "student"
if DADASETID == 1:
    DADASET_NAME = 'student'
if DADASETID == 2:
    DADASET_NAME = 'proff'

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from string import punctuation
from collections import Counter
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torchmetrics import F1Score as F1
# from torchmetrics.functional import f1, recall
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from google.colab import drive
# drive.mount('/content/gdrive')

"""## Скачивание и подготовка данных"""

train_data = pd.read_csv(f"content/train_{DADASET_NAME}.csv", sep='\t')
train_data.info()

val_data =  pd.read_csv(f"content/validate_{DADASET_NAME}.csv", sep='\t')
val_data.info()

vocab = Counter()
for text in tqdm(train_data.text):
    for symbol in text:
        if symbol is not "\n":
            vocab.update(list(symbol))
print('всего уникальных символов:', len(vocab))

#создаем словарь с индексами symbol2id, для спецсимвола паддинга дефолтный индекс - 0
symbol2id = {'PAD':0}

for symbol in tqdm(vocab):
    symbol2id[symbol] = len(symbol2id)

#обратный словарь для того, чтобы раскодировать последовательность
id2symbol = {i:symbol for symbol, i in symbol2id.items()}

label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
#train_data['label'] = np.where((train_data.label > 1), 1, train_data.label)

val_data['label'] = label_encoder.transform(val_data['label'])
#val_data['label'] = np.where((val_data.label > 1), 1, val_data.label)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE

"""## Dataset & DataLoader"""

class CharacterDataset(Dataset):

    def __init__(self, dataset, symbol2id, DEVICE):
        self.dataset = dataset['text'].values
        self.symbol2id = symbol2id
        self.length = dataset.shape[0]
        self.target = dataset['label'].values
        self.device = DEVICE

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        texts = self.dataset[index] 
        symbols = list(self.dataset[index])
        ids = torch.LongTensor([self.symbol2id[symbol] for symbol in symbols if symbol in self.symbol2id])
        y = self.target[index]
        return ids, y, texts

    def collate_fn(self, batch): #этот метод можно реализовывать и отдельно,
    # он понадобится для DataLoader во время итерации по батчам
      ids, y, texts = list(zip(*batch))
      padded_ids = pad_sequence(ids, batch_first=True).to(self.device)
      y = torch.LongTensor(y).to(self.device)
      return padded_ids, y, texts

train_dataset = CharacterDataset(train_data, symbol2id, DEVICE)
train_sampler = RandomSampler(train_dataset)
train_iterator = DataLoader(train_dataset, collate_fn = train_dataset.collate_fn, sampler=train_sampler, batch_size=64)

val_dataset = CharacterDataset(val_data, symbol2id, DEVICE)
val_sampler = SequentialSampler(val_dataset)
val_iterator = DataLoader(val_dataset, collate_fn = val_dataset.collate_fn, sampler=val_sampler, batch_size=64)

"""## CNN model"""

class CNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bigrams = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=2, padding='same')
        self.trigrams = nn.Conv1d(in_channels=embedding_dim, out_channels=80, kernel_size=3, padding='same')
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(in_features=180, out_features=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, word):
        #batch_size x seq_len
        embedded = self.embedding(word)
        #batch_size x seq_len x embedding_dim
        embedded = embedded.transpose(1,2)
        #batch_size x embedding_dim x seq_len
        feature_map_bigrams = self.dropout(self.pooling(self.relu(self.bigrams(embedded))))
        #batch_size x filter_count2 x seq_len* 
        feature_map_trigrams = self.dropout(self.pooling(self.relu(self.trigrams(embedded))))
        #batch_size x filter_count3 x seq_len*

        pooling1 = feature_map_bigrams.max(2)[0] 
        # batch_size x filter_count2
        pooling2 = feature_map_trigrams.max(2)[0]
        # batch_size x filter_count3
        concat = torch.cat((pooling1, pooling2), 1)
        # batch _size x (filter_count2 + filter_count3)
        logits = self.hidden(concat) 
        #logits = self.out(logits)      
        return logits

loss = nn.CrossEntropyLoss()

"""## training loop, логика обучения и валидации"""

def train(model, iterator, optimizer, criterion, df):
    epoch_loss = 0 # для подсчета среднего лосса на всех батчах
    model.train()  # ставим модель в обучение, явно указываем, что сейчас надо будет хранить градиенты у всех весов
    predictions = []
    acc = 0
    
    for i, (symbols, ys, texts) in enumerate(iterator): #итерируемся по батчам
        optimizer.zero_grad()  #обнуляем градиенты
        preds = model(symbols)
        loss = criterion(preds, ys) #считаем значение функции потерь
        loss.backward() #считаем градиенты  
        preds = preds.cpu().detach().numpy()  #прогоняем данные через модель
        predictions.extend(np.round(preds))
        optimizer.step() #обновляем веса 
        epoch_loss += loss.item() #сохраняем значение функции потерь
        if not (i + 1) % int(len(iterator)/5):
            print(f'Train loss: {epoch_loss/i}')
        acc += accuracy_score(ys.cpu().detach().numpy(), preds.argmax(axis=1))
    print(f'Train accuracy: {acc/len(iterator)}')

def evaluate(model, iterator, criterion, df):
    epoch_loss = 0
    accuracy = []
    predictions = []
    model.eval() 
    text = []
    labels = []
    with torch.no_grad():
        for i, (symbols, ys, texts) in enumerate(iterator):   
            preds = model(symbols)  # делаем предсказания на тесте
            loss = criterion(preds, ys)   # считаем значения функции ошибки для статистики  
            epoch_loss += loss.item()
            preds = preds.cpu().detach().numpy()
            predictions.extend(preds.argmax(axis=1))
            labels.extend(ys.cpu().detach().numpy())
            text.extend(list(texts))
            accuracy.append(accuracy_score(ys.cpu().detach().numpy(), preds.argmax(axis=1)))
        print(f'Val accuracy: {sum(accuracy)/ len(iterator)}') # возвращаем среднее значение по всей выборке
    return predictions, labels, text

"""## инициализируем модель, задаем оптимизатор и функцию потерь"""

model = CNN(len(symbol2id), 100)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss() 

# веса модели и значения лосса храним там же, где и все остальные тензоры
model = model.to(DEVICE)
criterion = criterion.to(DEVICE)

"""## запуск обучения"""

losses = []
losses_eval = []
f1s = []
f1s_eval = []

for i in tqdm(range(10)):
    print(f'\nstarting Epoch {i}')
    print('Training...')
    train(model, train_iterator, optimizer, criterion, train_data)
    print('\nEvaluating on test...')
    evaluate(model, val_iterator, criterion, val_data)

predictions, labels, texts = evaluate(model, val_iterator, criterion, val_data)
df = pd.DataFrame({'texts':texts, 'predictions':label_encoder.inverse_transform(predictions),
                   'labels':label_encoder.inverse_transform(labels)})

df.to_csv(f"content/CNN_{DADASET_NAME}.csv")

"""## работа с моделью"""

torch.save(model.state_dict(), f"content/CNN_model_{DADASET_NAME}.pth")