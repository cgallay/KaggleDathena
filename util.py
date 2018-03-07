import pickle
def save(obj, name):
    pickle.dump(obj, open( name, "wb" ))

def load(name):
    return pickle.load(open( name, "rb" ))
