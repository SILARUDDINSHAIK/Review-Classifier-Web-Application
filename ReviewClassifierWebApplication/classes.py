import torch
from flask import Flask, request, render_template
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow import one_hot
from transformers import BertModel, BertTokenizer
from transformers import TFBertForSequenceClassification
import torch.nn as nn

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.4)
        # self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.drop1 = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # output = self.relu(pooled_output)
        output = self.drop(pooled_output)
        output = self.out1(output)
        output = self.relu(output)
        output = self.drop1(output)
        return self.out(output)

    def save(self):
        model = torch.load('/Users/silaruddin/Desktop/Torchmodel')
        print('Model Loaded')
        return model
