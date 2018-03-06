# LSTM for sequence classification in the IMDB dataset
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard

def train_model(model_path='my_model.h5'):
    # Using keras to load the dataset with the top_words
    top_words = 100000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # Pad the sequence to the same length
    max_review_length = 1600        #TODO in constructor
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # Using embedding from Keras
    embedding_vecor_length = 300
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Convolution1D(32, 3, padding='same'))
    model.add(Convolution1D(16, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))

    # Log to tensorboard
    tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_split=0.2, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)

    # Evaluation on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model.save(model_path)


def apply(list_sentence,model):
    assert type(list_sentence) == list, 'The parameter must be a list'
    #assert type(list_sentence[0]) == str, 'arg must be a list of Sting'
    max_review_length = 1600
    l_pred = []

    #load the model (should be done in the constructor if a class is later created)
   
    word_index = imdb.get_word_index()

    for sent in list_sentence:
        words = sent.lower().split()
        new_data = np.array([word_index[word] if word in word_index else 0 for word in words])
        new_data = [i if i < 10000 else 0 for i in new_data]
        #print(new_data)
        l_pred.append(new_data)
    l_pred = sequence.pad_sequences(l_pred, maxlen=max_review_length)
    #print(model.predict_classes(l_pred))
    return model.predict_proba(l_pred)