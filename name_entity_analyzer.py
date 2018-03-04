import spacy

def apply(sentences):
    nlp = spacy.load('en_core_web_sm')
    for s in sentences:
        doc = nlp(s)
        for ent in doc2.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)