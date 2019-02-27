import pandas as pd
import os


def generate_directories(directory):
    
    try:
        os.mkdir(directory)
    except:
        pass

