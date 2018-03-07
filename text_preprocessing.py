import spacy
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import math
import warnings
import re

stopwords = set(['an', 'a', 'the', '-PRON-'])

class Preprocessor():
    def __init__(self, max_nb_words = 10000):
        #load spacy nlp pipline 
        self.nlp = spacy.load('en',  disable=['parser', 'ner'])
        self.max_nb_words = max_nb_words
        self.tokenizer = None

    #test that either dico is not None or self.dico is not None
    def preprocess(self, texts, dico=None): 
        """
        Preproces the multiple text by spliting it into token, remove stop words and encode it into a vector of integer
        Return: a tupple (encoded_texts, dict_words)
            encoded_texts:  ( for exemple ["I ate apple"] would return [[1, 3, 56, 78]]) 
            dict_words:  word -> integer (for exemple {I:1, eat:3, apple: 78 })        
        """
        cleaned_texts = []
        for text in tqdm(texts):
            cleaned_texts.append(self.preprocess_entity(text))
        if dico:
            tokenizer = tokenizer_from_dict(dico)
        else:
            tokenizer = self.tokenizer
        if self.tokenizer == None:
            raise Exception('no dictionary prodide please call fit_and_vectorize() first')
        
        return tokenizer.texts_to_sequences(cleaned_texts)

    def fit_and_vectorize(self, texts):       
        cleaned_texts = []
        for text in tqdm(texts):
            cleaned_texts.append(self.preprocess_entity(text))

        self.tokenizer = Tokenizer(num_words=self.max_nb_words)
        self.tokenizer.fit_on_texts(cleaned_texts)
        
        #TODO Warning the number of sentence returned might be bigger
        return (self.tokenizer.texts_to_sequences(cleaned_texts), self.tokenizer.word_index)

    def preprocess_entity(self, text):
        """Private methode"""
        processed_text = ''
        for sub_text in split_long_text(text):
            sub_text = " ".join(re.findall("([A-Za-z@?:!]*)",sub_text))
            doc = self.nlp(sub_text)
            for token in doc:
                if token.lemma_ not in stopwords:
                    processed_text+= token.lemma_ + ' '
        return processed_text


def tokenizer_from_dict(dico):
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
    