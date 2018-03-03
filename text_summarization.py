from nltk.tokenize import sent_tokenize, word_tokenize
def apply(text):
    """ Return: the imortant sentence of the text """
    sents = sent_tokenize(text)
    inter_sent = [sent for sent in sents if interesting_sent(sent)]
    return inter_sent
        

def interesting_sent(sent):
    return 'keppel' in sent.lower() or 'prudential' in sent.lower()

