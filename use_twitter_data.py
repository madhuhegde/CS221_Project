from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow as tf
import keras
import gc

# Reset Keras Session
#    sess = get_session()
#    clear_session()
#    sess.close()
#    sess = get_session()
#
#    try:
#        del classifier # this is from global space - change this as you need
#    except:
#        pass
#
#    print(gc.collect()) # if it's done something you should see a number being outputted
#
#    # use the same config as you used to create the session
#    config = tensorflow.ConfigProto()
#    config.gpu_options.per_process_gpu_memory_fraction = 1
#    config.gpu_options.visible_device_list = "0"
#    set_session(tensorflow.Session(config=config))
    
#reset_keras()


#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#
#from keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)
#
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#
#from keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import class_weight

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.regularizers import L1L2

train_df = pd.read_csv("./data/train.csv")
print("Train Data shape : ",train_df.shape)

train_new_df = pd.read_csv("./data/labeled_data.csv")
print("Train New Data shape : ",train_new_df.shape)

## Config values 
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=2019)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=2020)
print("Validation shape : ",val_df.shape)
print("Test shape : ",test_df.shape)
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
oov_tok = "OOV_TOK"
tokenizer = Tokenizer(num_words=max_features) # lower = False, oov_token=oov_tok)
tokenizer.fit_on_texts(list(train_X))

print(train_X.shape)
train_X = np.append(train_X, train_new_df["tweet"].values)
print(train_X.shape)
print(train_X[-5:])

#print(train_new_df.head())

train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentenc
train_X = pad_sequences(train_X, maxlen=maxlen, truncating='post')
val_X = pad_sequences(val_X, maxlen=maxlen, truncating='post')
test_X = pad_sequences(test_X, maxlen=maxlen, truncating='post')

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
test_y = test_df['target'].values


print(train_y.shape)
train_y = np.append(train_y, train_new_df["class"].values)
print(train_y.shape)
print(train_y[-5:])
train_y = np.logical_or(train_y == 0, train_y == 1).astype(int)
print(train_y.shape)
print(train_y[-5:])

EMBEDDING_FILE = './data/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


regularizer = L1L2(l1=0.0, l2=0.001)
#del model
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, trainable=True, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
#x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

print(model.summary())

# Define callback function if detailed log required

class History(Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_loss = []
        for keys in logs:
            print(keys) 

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
    #Store val_acc/loss per batch    
    def on_epoch_end(self, batch, logs={}):    
        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))
        
# Compute class_weights for imbalanced train set
def compute_class_weight(input_list):

    class_weights = class_weight.compute_class_weight('balanced', np.unique(input_list), input_list)
    return(class_weights)
        
#define callback functions
history = History()
callbacks = [history]        

class_weights = compute_class_weight(train_y)

class_weights[1] *= 10
print(class_weights)

model.fit(train_X, train_y, 
            batch_size=512, 
            epochs=1, 
            validation_data=(val_X, val_y), 
            class_weight = class_weights,
            callbacks= callbacks)
model.save('./output/model_glove_emb_add_data.h5')

from tensorflow.keras.models import Model, load_model

model =load_model('./output/model_glove_emb.h5')
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)
threshold = []
f1_array = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    threshold.append(thresh)
    f1_score = metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))
    f1_array.append(f1_score)
    print("F1 score at threshold {0} is {1}".format(thresh, f1_score))


pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)
threshold = []
f1_array = []
f1_max = 0
opt_thresh = 0
test_y = test_df['target'].values
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    threshold.append(thresh)
    f1_score = metrics.f1_score(test_y, (pred_test_y>thresh).astype(int))
    if(f1_score > f1_max):
        f1_max = f1_score
        opt_thresh = thresh
    f1_array.append(f1_score)
    print("F1 score at threshold {0} is {1}".format(thresh, f1_score))

pred = [int(a > opt_thresh) for a in pred_test_y]
