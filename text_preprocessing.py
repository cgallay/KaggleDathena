import spacy
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import math
import warnings
import re

"""
List of custom stop words.
"""
stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our','I',
               'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
               'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
               'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
               'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
               'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
               'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
               'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
               'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
               'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
               'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
               'just','should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                'ma', '-PRON-'])

"""
Class called to preprocess text. We use a class to load dictionnary and a model only once.
"""
class Preprocessor():
    def __init__(self, dico=None, max_nb_words = 10000):
        #load spacy nlp pipline 
        self.nlp = spacy.load('en',  disable=['parser', 'ner'])
        if dico == None:
            self.max_nb_words = max_nb_words
        else:
            self.max_nb_words = len(dico) + 1
        self.tokenizer = None
        self.dico = dico

    #test that either dico is not None or self.dico is not None
    def preprocess(self, texts): 
        """
        Preproces the multiple text by spliting it into token, remove stop words and encode it into a vector of integer
        Return: the encoded_texts:  ( for exemple ["I ate apple"] would return [[1, 3, 56, 78]]) 
        """
        cleaned_texts = []
        for text in tqdm(texts):
            cleaned_texts.append(self.preprocess_entity(text))
        if self.dico:
            tokenizer = tokenizer_from_dict(truncate_dict(self.dico, self.max_nb_words))
        else:
            tokenizer = self.tokenizer
        if tokenizer == None:
            raise Exception('no dictionary prodide please call fit_and_vectorize() first')
        
        return tokenizer.texts_to_sequences(cleaned_texts)

    def fit_and_vectorize(self, texts):   
        """
        Preprocess all texts with the pipeline preprocess tokenize lemmatize and stop word removal
        texts: array of text to fit and vectorize
        Return: phrased vectorized and dictionnary
        """
        cleaned_texts = []
        for text in tqdm(texts):
            cleaned_texts.append(self.preprocess_entity(text))

        self.tokenizer = Tokenizer(num_words=self.max_nb_words)
        self.tokenizer.fit_on_texts(cleaned_texts)

        return (self.tokenizer.texts_to_sequences(cleaned_texts),
            truncate_dict(self.tokenizer.word_index, self.max_nb_words))

    def preprocess_entity(self, text):
        """Private methode: clean text, lemmatize and remove stopWords"""
        processed_text = ''
        for sub_text in split_long_text(text):
            sub_text = " ".join(re.findall("([A-Za-z@?:!]*)",sub_text))
            doc = self.nlp(sub_text)
            for token in doc:
                if token.lemma_ not in stopwords:
                    processed_text+= token.lemma_ + ' '
        return processed_text

def truncate_dict(dico, nb_max):
    """Helper to truncate a dictionnary to the nb_max first word"""
    return {k:dico[k] for k in dico if dico[k] < nb_max}

def tokenizer_from_dict(dico):
    """ Define a new tokenizer in function of a tokenizer """
    tokenizer = Tokenizer()
    tokenizer.word_index = dico
    return tokenizer

def split_long_text(text, max_text_lenght = 10000):
    """When the text it too long spacy cause an out of memory error 
        So we use that methode to split the text into shorter text that spacy can handle"""
    split = text.split()
    text_lenght = len(split)
    sub_texts = []
    for i in range(math.ceil(text_lenght / max_text_lenght)):
        sub_texts.append(' '.join(split[i*max_text_lenght:(i+1)*max_text_lenght]))
    return sub_texts
    