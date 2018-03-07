import spacy
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import math
import warnings
import re

stopwords = set(['an', 'a', 'the', '-PRON-'])

def preprocess(text, nlp):
    text = " ".join(re.findall("([A-Za-z@?:!]*)",text))
    doc = nlp(text)
    processed_text = ''
    for token in doc:
        if token.lemma_ not in stopwords:
            processed_text+= token.lemma_ + ' '
    return processed_text

def preprocess_texts(texts, tokenizer):
    pass


def apply(texts, tokenizer = None):
    """
    Preproces the multiple text by spliting it into token, remove stop words and encode it into a vector of integer
    Return: a tupple (encoded_texts, dict_words)
        encoded_texts:  ( for exemple ["I ate apple"] would return [[1, 3, 56, 78]]) 
        dict_words:  word -> integer (for exemple {I:1, eat:3, apple: 78 })        
    """
    max_nb_words = 10000 #the maximal number of words that we consider (the size of the Dict)
    #nlp = spacy.load('en')  #loaded only once
    nlp = spacy.load('en', disable=['parser', 'ner'])   #disable to use less memory
    cleaned_texts = []
    for text in tqdm(texts):
        for sub_text in split_long_text(text):
            cleaned_texts.append(preprocess(sub_text, nlp))

    if tokenizer == None:
        tokenizer = Tokenizer(num_words=max_nb_words)
        tokenizer.fit_on_texts(cleaned_texts)
    
    #TODO Warning the number of sentence returned might be bigger
    return (tokenizer.texts_to_sequences(cleaned_texts), tokenizer.word_index)


def tokenizer_from_dict(dico):
    tokenizer = Tokenizer()
    tokenizer.word_index = dico
    return tokenizer

def generate_tokenizer(dico):
    warnings.warn('use tokenizer_from_dict instead', DeprecationWarning, stacklevel=2)
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
    