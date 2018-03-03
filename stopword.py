import nltk
from nltk.tokenize import word_tokenize

def removeStop(text):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
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
                'ma' ]
 
    word_tokens = word_tokenize(text)
 
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
 
    filtered_sentence = []
 
    for w in word_tokens:
        if w not in stop_words:
            if (w=="n\'t"):
                w="not"
            filtered_sentence.append(w)
    return filtered_sentence