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

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.utils import class_weight

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, Bidirectional, SpatialDropout1D
from keras.layers import BatchNormalization, Flatten
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer

from keras.initializers import *
from keras.optimizers import Nadam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
import tensorflow as tf
import os
import time
import gc
import re

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
         '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
           '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
            '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


train_df = pd.read_csv("./data/train.csv")
print("Train Data shape : ",train_df.shape)

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
#print("Test shape : ",test_df.shape)

train_new_df = pd.read_csv("./data/labeled_data.csv")
print("Train New Data shape : ",train_new_df.shape)
train_new_df["tweet"] = train_new_df["tweet"].apply(lambda x: clean_text(x))

## Config values 
embed_size = 300 # how big is each word vector
max_features = None # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

## fill up the missing values
X = train_df["question_text"].fillna("_na_").values
print(X.shape)
X = np.append(X, train_new_df["tweet"].values)
print(X.shape)
print(X[-5:])

## Get the target values
Y = train_df['target'].values
print(Y.shape)
Y_new = train_new_df["class"].values
print(Y_new.shape)
print(Y_new[-5:])
Y_new = np.logical_or(Y_new == 0, Y_new == 1).astype(int)
Y = np.append(Y, Y_new)
print(Y.shape)
print(Y[-5:])
#Y_test = test_df['target'].values

#X_test = test_df["question_text"].fillna("_na_").values
X, X_test, Y, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

tokenizer = Tokenizer(num_words=max_features, filters='') # lower=False)
tokenizer.fit_on_texts(list(X))

X = tokenizer.texts_to_sequences(X)
X_test = tokenizer.texts_to_sequences(X_test)

## Pad the sentences 
X = pad_sequences(X, maxlen=maxlen,truncating='post')
X_test = pad_sequences(X_test, maxlen=maxlen, truncating='post')


del train_df
gc.collect()


word_index = tokenizer.word_index
max_features = len(word_index)+1
def load_glove(word_index):
    EMBEDDING_FILE = './data/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
 
def load_para(word_index):
    EMBEDDING_FILE = './data/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


emb_matrix_1 = load_glove(word_index)
emb_matrix_2 = load_para(word_index)
embedding_matrix = np.add(0.7*emb_matrix_1, 0.3*emb_matrix_2)
del emb_matrix_1, emb_matrix_2
gc.collect()
np.shape(embedding_matrix)


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True, activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel', shape=(1, input_dim_capsule, self.num_capsule * self.dim_capsule),
                                    initializer='glorot_uniform', trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel', shape=(input_num_capsule, input_dim_capsule, self.num_capsule * self.dim_capsule),
                                    initializer='glorot_uniform', trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
      
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def model_final():
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    emb_out = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    sd_out = SpatialDropout1D(rate=0.2)(emb_out)
    lstm_out = Bidirectional(CuDNNLSTM(100, return_sequences=True, kernel_initializer=glorot_normal(seed=12345), 
                            recurrent_initializer=orthogonal(gain=1.0, seed=12345)))(sd_out)

    cap_out = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(lstm_out)
    fl_out = Flatten()(cap_out)

    d_out = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12345))(fl_out)
    drop_out = Dropout(0.12)(d_out)
    bn_out = BatchNormalization()(drop_out)

    final_out = Dense(1, activation="sigmoid")(bn_out)
    model = Model(inputs=inp, outputs=final_out)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(),metrics=['accuracy'])
    model.summary()
    return model

def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2

class History(Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_loss = []
        #for keys in logs:
            #print(keys) 

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
class_weights = compute_class_weight(Y)
print(class_weights)

#class_weights = [0.53290517, 80.97591407]
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
kfold_threshold = []
filepath="./output/weights_kfold_3.h5"
y_test = np.zeros((X_test.shape[0], ))
for i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):
    X_train, X_val, Y_train, Y_val = X[train_index], X[valid_index], Y[train_index], Y[valid_index]
  
    chk_point = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    #early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [chk_point, reduce_lr, history]
    model = model_final()
    if i == 0:
        print(model.summary()) 
    model.fit(X_train, Y_train, batch_size=512, epochs=5, validation_data=(X_val, Y_val), verbose=2, 
                          class_weight=class_weights, callbacks=callbacks,)
    #model.load_weights(filepath)
    y_pred = model.predict([X_val], batch_size=1024, verbose=2)
    y_test += np.squeeze(model.predict([X_test], batch_size=1024, verbose=2))/5
    
    #select the threshold that maximizes y_pred
    f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))
    print('Optimal F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
    kfold_threshold.append(threshold)

y_test = y_test.reshape((-1, 1))
avg_threshold = np.mean(kfold_threshold)
pred_test_y = (y_test>avg_threshold).astype(int)
f1_score = metrics.f1_score(Y_test, pred_test_y)
print("F1 Score on Final Test Set: ", f1_score)
