from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
import spacy

def apply(text, interest):
    """ Return: the imortant sentence of the text """
    #sents = sent_tokenize(text)
    #inter_sent = [sent for sent in sents if interesting_sent(sent)]
    return get_surroundning(text, interest, 20)

def interesting_sent(sent):
    #NER = ne_chunk(pos_tag(word_tokenize(sent)))
    return 'keppel' in sent.lower() or 'prudential' in sent.lower()

def get_surroundning(text, word, nb_words):
    nlp = spacy.load('en_core_web_sm') #TODO speed up by putting that in an object constuctor
    doc = nlp(text) #apply the preprcessing pipline
    sur = [doc[(i-nb_words):i+nb_words+1].text for (i, w) in enumerate(doc) if w.text.lower() == word.lower()]
    return sur
