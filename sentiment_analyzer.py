# LSTM for sequence classification in the IMDB dataset
import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, MaxPooling1D, Subtract, Dot, Reshape
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
import text_preprocessing
import util
import numpy as np
from nltk.corpus import sentiwordnet as swn

datasets = ['Amazon', 'IMDB']
nb_rand_sample = 1000 #this is the number of pairs that we compare for each batch
weights_corp = np.load(open('safe/embeddings.np', 'rb'))
sigma_reg_emb = 0.1    #hyper parameter for embedding regularization term

class SentimentAnalyzer():
    nb_lines_amazon = 500000    #nb of training to read from the Amazon review dataset
    max_sent_length = 1600  #Maximent size of a sentence that the analyzer can fit
    def __init__(self, model_path=None):
        word_index = util.load('safe/vocab_gensim.p')
        self.preprocessor = text_preprocessing.Preprocessor(word_index, 100000)
        if model_path:
            with keras.utils.CustomObjectScope({'embbeding_reg':get_reg}):
                self.model = keras.models.load_model(model_path)
    
    def train(self, dataset='Amazon', model_path='models/my_model.h5',epochs=1, top_words=100000):
        assert dataset in datasets, 'Dataset should be in that list ' + str(datasets)
        if dataset == 'Amazon':
            X_train, y_train = load_dataset('dataset/amazonreviews/data', self.nb_lines_amazon)
        else:
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
            #as the preprocessing is not the same we dont support this dataset anymore
            raise Exception('IMDB dataset is not supported anymore for training')
        
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_sent_length)  
        
        model = self.generateModel()

        # Log to tensorboard
        tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
        model.fit(X_train, y_train , validation_split=0.2, epochs=epochs, callbacks=[tensorBoardCallback], batch_size=64)

        model.save(model_path)

    def generateModel(self):
        top_words = weights_corp.shape[0]
        embedding_vecor_length = weights_corp.shape[1]

        #define our own Keras model
        main_input = Input(shape=(self.max_sent_length,), dtype='int32', name='word_input')
        embedding = Embedding(output_dim=embedding_vecor_length, input_dim=top_words, embeddings_regularizer=embbeding_reg) #share embeding layer

        out_emb = embedding(main_input)
        conv1 = Convolution1D(64, 3, padding='same')(out_emb)
        conv2 = Convolution1D(32, 3, padding='same')(conv1)
        conv3 = Convolution1D(16, 3, padding='same')(conv2)
        flat_layer = Flatten()(conv3)
        drop1 = Dropout(0.2)(flat_layer)
        dense1 = Dense(180,activation='sigmoid')(drop1)
        drop2 = Dropout(0.2)(dense1)
        sentiment_output = Dense(1,activation='sigmoid', name='sentiment_output')(drop2) 

        model = Model(inputs=main_input, outputs=sentiment_output)
     
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def predict(self, list_sentence):
        """the list_sentence provided is of raw string data"""
        assert type(list_sentence) == list, 'The parameter must be a list'
       
        new_data = self.preprocessor.preprocess(list_sentence)
        
        l_pred = sequence.pad_sequences(new_data, maxlen=self.max_sent_length)

        return self.model.predict(l_pred)

#Unused 
class sentiwordnetAnalyzer():
    nb_lines_amazon = 100 #dont need to be as big as for the CNN as this model is much simpler
    def __init__(self, model_path=None):
        word_index = util.load('safe/vocab_gensim.p') #util.load('safe/dico.p') #imdb.get_word_index()
        self.preprocessor = text_preprocessing.Preprocessor(word_index)
        raise Exception("Class is not fully implemented yet...")

    def train(self, dataset = "Amazon", top_words=10000):
        """Once the CNN is train we can train a model on top of this one.
        A simple logistic regression that have also access to the senti wordnet dico    
        """
        assert dataset in datasets, 'Dataset should be in that list ' + str(datasets)
        if dataset == 'Amazon':
            X_train, y_train = load_dataset('dataset/amazonreviews/data', self.nb_lines_amazon)
        else:
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
            raise Exception('Dead code... This should be retest again')

               

def read_line(line):
    """This methode read one line of the Amazon dataset and extract the corresponding label and text feature
    Return:
        (X, y): a tupple consisting of the text feature and its label y=0 if negatif and 1 if possitif 
    """
    label = line[0:11]
    text = line[11:]
    y = 1 if label == '__label__2 ' else 0
    return text, y

def load_dataset(fname, nb_lines):
    """Load the Amazon dataset if not already present on disc"""
    import os.path
    if os.path.isfile('safe/Amazon-'+str(nb_lines)+'.p'):
        return util.load('safe/Amazon-'+str(nb_lines)+'.p')
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
    preprocessor = text_preprocessing.Preprocessor(dico=dico)
    X = preprocessor.preprocess(X)
    #save the loaded dataset in a pickle for speeding up next run
    util.save((X,y), 'safe/Amazon-'+str(nb_lines)+'.p')
    return (X, y)

def sampling(args):
    '''Sample at random one vector'''
    #TODO not fix the maxval param
    return K.random_uniform(shape=[], minval=1, maxval=99999, dtype='int32')

def get_reg():
    """Return the function use for embedding regularization, this function is needed for keras to reload the save model"""
    return embbeding_reg

def embbeding_reg(weight_matrix):
    """Penalize the weight of the embedding such that two similar words in the corpus embedding spaced trained using gensim stay similar in the new emebedding space"""
    shape = weight_matrix.shape

    #Sample at random nb_rand_sample of words pairs
    z1 = tf.random_uniform([nb_rand_sample], minval = 0, maxval = shape[0] - 1, dtype = tf.int32)
    z2 = tf.random_uniform([nb_rand_sample], minval = 0, maxval = shape[0] - 1, dtype = tf.int32)
    
    #Select the corresponding vector in the embedding space
    vec_1 = tf.gather(weight_matrix, z1)
    vec_2 = tf.gather(weight_matrix, z2)

    #compute the cosine similarity between each pairs
    vec_1_norm = K.l2_normalize(vec_1, 1)
    vec_2_norm = K.l2_normalize(vec_2, 1)
    cosine_sim = K.sum(vec_1_norm * vec_2_norm, axis=1)
    
    #initialize the pretrain embedding space
    weight_objective = K.constant(weights_corp)

    #Select the corresponding vector in the embedding space
    vec_1_corp = tf.gather(weight_objective, z1)
    vec_2_corp = tf.gather(weight_objective, z2)

    #compute the cosine similarity between each pairs
    vec_1_corp_norm = K.l2_normalize(vec_1_corp, 1)
    vec_2_corp_norm = K.l2_normalize(vec_2_corp, 1)
    cosine_sim_corp = K.sum(vec_1_corp_norm * vec_2_corp_norm, axis=1) #scalar product

    #Compute the Mean Square Error
    diff = cosine_sim - cosine_sim_corp
    square = K.square(diff) #Square error
    MSE = K.mean(square)

    return sigma_reg_emb * MSE