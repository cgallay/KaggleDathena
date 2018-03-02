def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def lemmize(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    finalStemmed = []
    for w in text : 
        allInfo = w
        word = allInfo[0]
        tag = allInfo[1]
        stemmedWord = ""
        try:
            stemmedWord = wordnet_lemmatizer.lemmatize(word,get_wordnet_pos(tag))
            finalStemmed.append((stemmedWord,w[1][0]))
        except Exception:
            finalStemmed.append((w[0],w[1][0]))     
    return finalStemmed