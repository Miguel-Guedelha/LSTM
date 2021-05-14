import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import random
import os
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
from lxml import etree
import re
from emoji import UNICODE_EMOJI
import pandas as pd
import torch
from sympy.physics.units import cm
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import keras
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(r, pathlist):
    data = r.read().split("\n")
    idk = []  # id
    spreader = []  # yes or no
    for line in data:
        l = line.split(":::")
        if len(l) > 1:
            idk.append(l[0])
            spreader.append(l[1])

    meta_data = pd.DataFrame()
    meta_data["ID"] = idk
    meta_data["spreader"] = spreader

    # Reading in and concatenating English tweets

    ids = []
    x_raw = []

    for path in pathlist:
        # iterate files
        head, tail = os.path.split(path)
        t = tail.split(".")
        author = t[0]
        ids.append(author)
        path_in_str = str(path)
        tree = ET.parse(path_in_str)
        root = tree.getroot()

        for child in root:
            xi = []

            for ch in child:
                xi.append(ch.text)
            content = ' '.join(xi)

            x_raw.append(content)

    text_data = pd.DataFrame()
    text_data["ID"] = ids
    text_data["Tweets"] = x_raw

    # Merging meta data and text data to one dataframe
    en_data = pd.merge(meta_data, text_data, how='inner', on='ID')

    feed_list = en_data["Tweets"].tolist()
    return feed_list, en_data


def cleaning_v1(tweet_lista):
    cleaned_feed_v1 = []
    for feed in tweet_lista:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        cleaned_feed_v1.append(feed)
    return cleaned_feed_v1


if __name__ == '__main__':
    print("dataset")
    r = open('fakevsreal/train/truth.txt',"r")
    pathlist =Path('fakevsreal/train').glob('**/*.xml')
    feed_list, en_data = preprocess(r, pathlist)
    en_data["Tweets"] = cleaning_v1(feed_list)
    x = feed_list
    y = en_data["spreader"]
    #x, x_test, y, y_test = train_test_split(feed_list, y, random_state=0, test_size=0.5, shuffle=True)
    max_words = 5000
    max_len = 600
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(x)
    train_seq = tok.texts_to_sequences(x)
    #test_seq = tok.texts_to_sequences(x_test)

    train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
    #test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)
    y = y.astype(np.int32)
    #y_test = y_test.astype(np.int32)
    print(train_seq_mat.shape)





inputs = Input(name='inputs',shape=[max_len])

layer = Embedding(max_words+1,128,input_length=max_len)(inputs)
layer = LSTM(128)(layer)
layer = Dense(128,activation="relu",name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1,activation="softmax",name="FC2")(layer)
model = Model(inputs=inputs,outputs=layer)
#keras.layers.recurrent.LSTM(64, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
model.summary()
model.compile(loss="binary_crossentropy",optimizer=RMSprop(),metrics=["accuracy"])
model_fit = model.fit(train_seq_mat,y,batch_size=20,epochs=200
                     )
model.get_config()


# Evaluation Function

def evaluate(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in test_loader:
            output = model(titletext, titletext_len)
            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])

