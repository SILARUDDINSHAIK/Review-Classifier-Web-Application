import ssl
import nltk
import numpy as np
import tensorflow as tf
import torch
import torch as torch
import torch.nn as nn
from flask import Flask, request, render_template, flash
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow import one_hot
from transformers import BertModel, BertTokenizer
from transformers import TFBertForSequenceClassification
from classes import SentimentClassifier
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# declare constants
HOST = '0.0.0.0'
PORT = 8888

# initialize flask application
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
# Read model to keep it ready all the time
# model = MyModel('./ml_model/trained_weights.pth', 'cpu')
CLASS_MAPPING = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)


@app.route('/')
def home():
    return render_template("home.html")


# @app.route('/predict', methods=['GET','POST'])

# @app.route('/', methods=["POST"])
# def predict():
# from transformers import BertTokenizer, TFBertForSequenceClassification
#
# return "Check what is wrong"

@app.route('/test_link/')
def test_link():
    return render_template('test_link.html')


def get_corpus(review):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    corpus = []
    voc_size = 300
    # review = review.lower()
    # review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    onehot_repr1 = [one_hot(words, voc_size) for words in corpus]
    # onehot_repr1
    return onehot_repr1


def get_embed(onehot_repr1):
    sent_length = 300
    embedded_docs = pad_sequences(onehot_repr1, padding='pre', maxlen=sent_length)
    X_final = np.array(embedded_docs)
    return X_final


@app.route('/next/', methods=["POST"])
def predict3():
    print('In predict-3')
    max_features = 20000
    max_text_len = 400
    embedding_dim = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    filters = 128
    kernel_size = 5
    hidden_dims = 24
    #tokenizer.fit_on_texts()
    #word_index = tokenizer.word_index
    #tokenized = tokenizer.texts_to_sequences()
    # X_train = sequence.pad_sequences(tokenized, maxlen=max_text_len, padding=padding_type, truncating=trunc_type)
    toxic_model = tf.keras.models.load_model("/Users/silaruddin/Desktop/SecureNLP/Toxic/toxic_classification.h5")
    tokenizer = text.Tokenizer(max_features, oov_token=oov_tok)
    comment = request.form.get('comment_text')
    print(comment)
    tokenized = tokenizer.texts_to_sequences(comment)
    print(tokenized)
    X_test = sequence.pad_sequences(tokenized, maxlen=max_text_len, padding=padding_type, truncating=trunc_type)
    y_test1 = toxic_model.predict(X_test, verbose=1, batch_size=128)
    print(y_test1)
    for x in y_test1:
        print(x)
        if x > .5:
            return 'Postive Comment'
        else:
            return 'Negative Comment'


@app.route('/forward/', methods=["POST"])
def predict2():
    model2 = tf.keras.models.load_model("/Users/silaruddin/Desktop/SecureNLP/Yelp/YelpWCNN")
    review = request.form.get('comment_text')
    #print(review)
    comment = review.lower().split()
    doc1 = set(comment)
    doc1 = sorted(comment)
    integer_encoded = []
    for i in comment:
        try:
            v = np.where(np.array(doc1) == i)[0][0]
            _create_unverified_https_context = ssl._create_unverified_context
            integer_encoded.append(v)
        except AttributeError:
            pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    def get_vec(len_doc, word):
        empty_vector = [0] * len_doc
        vect = 0  # nltk.download('stopwords')
        find = np.where(np.array(doc1) == word)[0][0]  # from nltk.stem.porter import PorterStemmer
        empty_vector[find] = 1  # ps = PorterStemmer()
        return empty_vector  # corpus = []

    def get_matrix(doc1):

        mat = []

        len_doc = len(doc1)

        for i in comment:
            # voc_size = 20
            vec = get_vec(len_doc, i)
            # review = review.lower()
            mat.append(vec)
            # review = review.split()

        return np.asarray(mat)

    onehot_repr1 = get_matrix(doc1)

    sent_length = 300
    embedded_docs = pad_sequences(onehot_repr1, padding='pre', maxlen=sent_length)
    X_final1 = np.array(embedded_docs)
    y_test1 = model2.predict(X_final1, verbose=1, batch_size=128)
    msg = ''
    for x in y_test1:
        if x > .5:
            flash(msg + 'Positive Review')
        else:
            flash(msg + 'Negative Review')
        return render_template("login.html",review1=review)

@app.route('/forward2/', methods=["POST"])
def predictionsbert():
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    class_name = ['negative', 'positive']
    # model = torch.load('/Users/silaruddin/Desktop/Torchmodel')
    # print('Model Loaded')
    obj = SentimentClassifier(len(class_name))
    model = obj.save()
    print('Model Loaded')
    model.eval()
    # model = SentimentClassifier(len(class_name))
    model = model.to(torch.device)
    MAX_LEN = 120
    review = request.form.get('new_text')
    print(review)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # review_text = "You, sir, are my grinder. whatsoever happen you think what paginate that's on?"
    encoded_review = tokenizer.encode_plus(
        review,
        max_length=MAX_LEN,
        add_special_tokens=True,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    # return f'Review text: {review}'
    return f'Sentiment  : {class_name[prediction]}'


if __name__ == '__main__':
    # run web server
    # test_fxn()
    app.run(HOST='0.0.0.0',
            PORT=8888)
