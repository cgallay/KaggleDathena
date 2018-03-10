# LSTM for sequence classification in the IMDB dataset
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
import text_preprocessing
import util
import numpy as np

class SentimentAnalyzer():
    datasets = ['Amazon', 'IMDB']
    nb_lines_amazon = 100
    max_sent_length = 1600
    def __init__(self, model_path=None):
        word_index = util.load('safe/vocab_gensim.p') #util.load('safe/dico.p') #imdb.get_word_index()
        self.preprocessor = text_preprocessing.Preprocessor(word_index)
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            #model need to be trained
            pass
    def train(self, dataset='Amazon', model_path='models/my_model.h5',epochs=1, top_words=10000):
        assert dataset in self.datasets, 'Dataset should be in that list ' + str(self.datasets)
        if dataset == 'Amazon':
            X_train, y_train = load_dataset('dataset/amazonreviews/data', self.nb_lines_amazon)
        else:
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
            raise Exception('Dead code... This should be retest again')
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_sent_length)
        #X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)      
        
        #load the pretrained embeddings
        weights = np.load(open('safe/embeddings.np', 'rb'))
        top_words = weights.shape[0]
        embedding_vecor_length = weights.shape[1]
        #print(f'weight loaded are of cardinality {top_words} and size {embedding_vecor_length}')

        # Using embedding from Keras
        model = Sequential()
        model.add(Embedding(top_words, embedding_vecor_length, weights=[weights], 
            trainable=False, input_length=self.max_sent_length))

        # Convolutional model (3x conv, flatten, 2x dense)
        model.add(Convolution1D(64, 3, padding='same'))
        model.add(Convolution1D(32, 3, padding='same'))
        model.add(Convolution1D(16, 3, padding='same'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(180,activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='sigmoid'))

        # Log to tensorboard
        tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, callbacks=[tensorBoardCallback], batch_size=64)

        # Evaluation on the test set
        #scores = model.evaluate(X_test, y_test, verbose=0)
        #print("Accuracy: %.2f%%" % (scores[1]*100))
        model.save(model_path)
    
    def predict(self, list_sentence):
        """the list_sentence provided is of raw string data"""
        assert type(list_sentence) == list, 'The parameter must be a list'
        #assert type(list_sentence[0]) == str, 'arg must be a list of Sting'
       
        #TODO need to be checked if the model was trained using IMDB or Amazon
        #For the moment let's assume Amazon
        new_data = self.preprocessor.preprocess(list_sentence)
        
        l_pred = sequence.pad_sequences(new_data, maxlen=self.max_sent_length)
        #print(model.predict_classes(l_pred))
        return self.model.predict_proba(l_pred)

def read_line(line):
    label = line[0:11]
    text = line[11:]
    y = 1 if label == '__label__2 ' else 0
    return text, y

def load_dataset(fname, nb_lines):
    """Load the Amazon dataset"""
    count = 1
    X = []
    y = []
    with open(fname) as f:
        for line in f:
            text, label = read_line(line)
            #print((label, text))
            X.append(text)
            y.append(label)
            if count >= nb_lines:
                break
            count+=1

    #load pretrained dictonary
    dico = util.load('safe/vocab_gensim.p')
    preprocessor = text_preprocessing.Preprocessor(dico=dico) #TODO use the good number of max_word
    X = preprocessor.preprocess(X)
    return (X, y)
