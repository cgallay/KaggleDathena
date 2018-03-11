from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def apply(text, interest, top_k=5):
    """
    Return and tuple of list of tuple.
        The first list contains all the sentence about the interest and there corresponding weigth in the document
        The second list contains the top_k sentences of the document
    """
    LANGUAGE = "english"
    parser =  PlaintextParser(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    sent_importance = summarizer.rate_sentences(parser.document)
    interesting_sent = []
    for sent in sent_importance:
        if interest.lower() in sent._text.lower():
            interesting_sent.append((sent._text, sent_importance[sent]))
    top_sent = summarizer(parser.document, top_k)
    top_sent = [(s._text, sent_importance[s]) for s in top_sent]
    return (interesting_sent, top_sent)