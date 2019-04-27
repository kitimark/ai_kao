import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import re
import glob
from keras import callbacks
from tensorflow.python.lib.io import file_io
import argparse

import csv
import deepcut
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# def model(max_feature,embedding_dim,lstm_unit,input_dim):

#     model = sequential()
#     model.add(embedding(max_feature, embedding_dim, input_length=input_dim))
#     model.add(spatialdropout1d(0.4))
#     model.add(lstm(lstm_unit, dropout=0.2, recurrent_dropout=0.2))
#     model.add(dense(2, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
    # return model

# def preparing_data(data,max_fatures=3000):
#     data = data[['CONTENT', 'CLASS']]
#     data['CONTENT'] = data['CONTENT'].apply(lambda x: x.lower())
#     data['CONTENT'] = data['CONTENT'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

#     tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#     tokenizer.fit_on_texts(data['CONTENT'].values)
#     X = tokenizer.texts_to_sequences(data['CONTENT'].values)
#     X = pad_sequences(X)
#     Y = pd.get_dummies(data['CLASS']).values

#     return X,Y

def read_multiple_csv(path):
    with file_io.FileIO(path, mode='r') as file:
        df = pd.read_csv(file, index_col=None, header=0)
    return df

def main(job_dir,**args):
    # #tuning parameter
    # max_feature = 3000
    # embedding_dim = 128
    # lstm_unit = 128
    # batch_size = 32
    # epoch = 7

    csv_folder = job_dir + 'data/dataset.csv'
    # logs_path = job_dir + 'logs/tensorboard'

    # #read data set from csv folder
    # df = read_multiple_csv(csv_folder)

    # #preparing data for training data
    # X,Y = preparing_data(df,max_feature)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # ## Adding the callback for TensorBoard and History
    # tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    # #Initial model
    # model = Model(max_feature,embedding_dim,lstm_unit,X.shape[1])

    # ##fitting the model
    # model.fit(x=X_train, y=Y_train, epochs=epoch, verbose=1, batch_size=batch_size, callbacks=[tensorboard],validation_data = (X_test, Y_test))

    #------------------------- Read data ------------------------------
    # file = open(csv_folder, 'r',encoding = 'utf-8-sig')
    # data = list(csv.reader(file))
    data = read_multiple_csv(csv_folder)
    shuffle(data)

    # for d in data:
    #     print(d)

    LABEL_NAMES = { "H" : 0, "M" : 1, "P" : 2}
    labels = [LABEL_NAMES[d[0]] for d in data]
    sentences = [d[1] for d in data]

    words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
    # for sentence in words:
    #     print(sentence)

    #------------------- Extract bag-of-words -------------------------
    vocab = set([w for s in words for w in s])

    print('Vocab size = '+str(len(vocab)))

    bag_of_words = np.zeros((len(words),len(vocab)))
    for i in range(0,len(words)):
        count = 0
        for j in range(0,len(words[i])):
            k = 0
            for w in vocab:
                if(words[i][j] == w):
                    bag_of_words[i][k] = bag_of_words[i][k]+1
                    count = count+1
                k = k+1
        bag_of_words[i] = bag_of_words[i]/count

    print(bag_of_words[0])

    #--------------- Create feedforward neural network-----------------
    inputLayer = Input(shape=(len(vocab),))
    h1 = Dense(1024, activation='tanh')(inputLayer)
    h2 = Dense(512, activation='tanh')(h1)
    h3 = Dense(128, activation='tanh')(h2)
    # l1 = Dense(64, activation='tanh')(inputLayer)
    # l2 = Dense(64, activation='tanh')(l1)
    outputLayer = Dense(3, activation='softmax')(h3)
    model = Model(inputs=inputLayer, outputs=outputLayer)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #----------------------- Train neural network-----------------------
    history = model.fit(bag_of_words, to_categorical(labels), epochs=300, batch_size=50, validation_split = 0.2)


    #-------------------------- Evaluation------------------------------
    y_pred = model.predict(bag_of_words[240:,:])

    # Save model.h5 on to google storage
    model.save('model.h5')
    with file_io.fileio('model.h5', mode='rb') as input_f:
        with file_io.fileio(job_dir + 'models/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

    cm = confusion_matrix(labels[240:], y_pred.argmax(axis=1)) 
    print('Confusion Matrix')
    print(cm)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.savefig('graph.png')
    with file_io.fileio('graph.png', mode='rb') as input_f:
        with file_io.fileio(job_dir + 'models/graph.png', mode='w+') as output_f:
            output_f.write(input_f.read())

##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)