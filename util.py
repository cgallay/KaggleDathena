import pickle

def save(obj, name):
    """
    args:
        obj: Object to save on the disc
        name: path where to save the object
    """
    pickle.dump(obj, open( name, "wb" ))

def load(name):
    """
    args:
        name: path where the object is located
    Return:
        Object pickle loaded from pathe name
    """
    return pickle.load(open( name, "rb" ))
