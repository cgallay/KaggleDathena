# LSTM for sequence classification in the IMDB dataset
import keras
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
import tensorflow as tf

datasets = ['Amazon', 'IMDB']
nb_comparaison = 100 #this is the number of pairs that we compare for each batch

class SentimentAnalyzer():
    nb_lines_amazon = 100
    max_sent_length = 1600
    def __init__(self, model_path=None):
        word_index = util.load('safe/vocab_gensim.p') #util.load('safe/dico.p') #imdb.get_word_index()
        self.preprocessor = text_preprocessing.Preprocessor(word_index, 100000)
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            #model need to be trained
            pass
    def train(self, dataset='Amazon', model_path='models/my_model.h5',epochs=1, top_words=100000, emb_trainable=False):
        assert dataset in datasets, 'Dataset should be in that list ' + str(datasets)
        if dataset == 'Amazon':
            X_train, y_train = load_dataset('dataset/amazonreviews/data', self.nb_lines_amazon)
        else:
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
            raise Exception('Dead code... This should be retest again')
        
        #print(X_train[0:10])
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_sent_length)
        #X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)      
        
        model_2 = self.generateModel()
        # Log to tensorboard
        tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
        model_2.fit(X_train, y_train , validation_split=0.2, epochs=epochs, callbacks=[tensorBoardCallback], batch_size=64)

        # Evaluation on the test set
        #scores = model.evaluate(X_test, y_test, verbose=0)
        #print("Accuracy: %.2f%%" % (scores[1]*100))
        model_2.save(model_path)

    def generateModel(self):
        weights = np.load(open('safe/embeddings.np', 'rb'))
        top_words = weights.shape[0]
        embedding_vecor_length = weights.shape[1]

        #define our own Keras model
        main_input = Input(shape=(self.max_sent_length,), dtype='int32', name='word_input')
        embedding = Embedding(output_dim=embedding_vecor_length, input_dim=top_words, embeddings_regularizer=embbeding_reg)# share embeding

        out_emb = embedding(main_input)
        conv1 = Convolution1D(64, 3, padding='same')(out_emb)
        conv2 = Convolution1D(32, 3, padding='same')(conv1)
        conv3 = Convolution1D(16, 3, padding='same')(conv2)
        flat_layer = Flatten()(conv3)
        drop1 = Dropout(0.2)(flat_layer)
        dense1 = Dense(180,activation='sigmoid')(drop1)
        drop2 = Dropout(0.2)(dense1)
        sentiment_output = Dense(1,activation='sigmoid', name='sentiment_output')(drop2) 
        #fake_input = Input(batch_shape=(100,), name='fake_input')
        #z1 = Lambda(sampling, output_shape=[], name='rand_1')([fake_input]) #select random pair of embeddings
        #z2 = Lambda(sampling, output_shape=[], name='rand_2')([fake_input]) #select random pair of embeddings
        #load the pretrained embeddings
        #embedding_corpus = Embedding(output_dim=embedding_vecor_length, input_dim=top_words,
        #    weights=[weights], trainable=False, name='corpus_emb')
        #emb_1 = embedding(z1)
        #emb_2 = embedding(z2)
        #emb_corp_1 = embedding_corpus(z1)
        #emb_corp_2 = embedding_corpus(z2)

        #sent_sim = Dot(axes=1, normalize=True, name='sentence_sim')([emb_1, emb_2])
        #corpus_sim = Dot(axes=1, normalize=True, name='corpus_sim')([emb_corp_1, emb_corp_2])

        #diff = Subtract(name='diff')([sent_sim, corpus_sim])
        #square_diff_var = Lambda(square_diff, name='output_reg')([diff])

        #define our Model with two output
        model_2 = Model(inputs=main_input, outputs=sentiment_output)
     
        model_2.compile(loss=model_loss_with_reg, optimizer='adam', metrics=['accuracy'])
        return model_2

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

class sentiwordnetAnalyzer():
    nb_lines_amazon = 100 #dont need to be as big as for the CNN as this model is much simpler
    def __init__(self, model_path=None):
        word_index = util.load('safe/vocab_gensim.p') #util.load('safe/dico.p') #imdb.get_word_index()
        self.preprocessor = text_preprocessing.Preprocessor(word_index)

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
    nb_comparaison = 100 #this is the number of pairs that we compare for each batch
    label = line[0:11]
    text = line[11:]
    y = 1 if label == '__label__2 ' else 0
    return text, y

def load_dataset(fname, nb_lines):
    """Load the Amazon dataset"""
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
    preprocessor = text_preprocessing.Preprocessor(dico=dico) #TODO use the good number of max_word
    X = preprocessor.preprocess(X)
    util.save((X,y), 'safe/Amazon-'+str(nb_lines)+'.p')
    return (X, y)

#keras loss function
def model_loss_with_reg(y_true, y_pred):
    cross_entropy = keras.losses.binary_crossentropy(y_true, y_pred)
    return cross_entropy
def loss_MSE_sim(y_true, y_pred):
    #y_true is not usefull
    return K.mean(y_pred)


def sampling(args):
    '''Sample at random one vector'''
    #TODO not fix the maxval param
    return K.random_uniform(shape=[], minval=1, maxval=99999, dtype='int32')

def square_diff(args):
    diff = args
    #diff = keras.layers.subtract([sent_sim, corpus_sim])
    return K.square(diff)

def wordCorpuLookup(args):
    indexs, weight = args
    return weight[indexs]

weight = np.load(open('safe/embeddings.np', 'rb'))
nb_rand_sample = 1000
lambda_reg_emb = 0.1

def embbeding_reg(weight_matrix):
    shape = weight_matrix.shape
    #print(type(shape))
    #z1 = np.random.randint(0,shape[0], 100)
    z1 = tf.random_uniform([nb_rand_sample], minval = 0, maxval = shape[0] - 1, dtype = tf.int32)
    #z2 = np.random.randint(0,shape[0], 100)
    z2 = tf.random_uniform([nb_rand_sample], minval = 0, maxval = shape[0] - 1, dtype = tf.int32)

    vectors1 = tf.gather(weight_matrix, z1, name='random_gather')
    vectors2 = tf.gather(weight_matrix, z2)
    vectors_1 = K.l2_normalize(vectors1, 1)
    vectors_2 = K.l2_normalize(vectors2, 1)

    vec = K.sum(vectors_1 * vectors_2, axis=1) #scalar product
    
    weight_tensor = K.constant(weight)
    vectors1_corp = tf.gather(weight_tensor, z1)
    vectors2_corp = tf.gather(weight_tensor, z2)
    vectors_1_corp = K.l2_normalize(vectors1_corp, 1)
    vectors_2_corp = K.l2_normalize(vectors2_corp, 1)
    vec_corp = K.sum(vectors_1_corp * vectors_2_corp, axis=1) #scalar product

    diff = vec - vec_corp
    square = K.square(diff) #Square error

    return lambda_reg_emb * K.mean(square) # MSE