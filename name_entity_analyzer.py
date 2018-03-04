import spacy

def apply(sentences,company):
    nlp = spacy.load('en_core_web_sm')
    finalPhrases = []
    for s in sentences:
        doc = nlp(s)
        for ent in doc.ents:
            print(ent.text , " ", ent.label_)
            if(company in ent.text.lower()):
                print(ent.text.lower())
                if(ent.label_ != "PERSON"):
                    print(s)
                    finalPhrases.append(s)
    return finalPhrases