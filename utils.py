import pandas as pd
import os
import pickle


def generate_directories(directory):
    
    try:
        os.mkdir(directory)
    except:
        pass

def dump(obj, filename):
    """ dump any object to a pickle filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    
def load(filename):
    """ Load any object from a pickle filename
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj