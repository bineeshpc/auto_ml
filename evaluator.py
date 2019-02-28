#! /usr/bin/env python


"""
Do K fold cross validation
Select a winning model
Save it for later use
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logger
import config_parser
import utils
import argparse
import pandas as pd
import numpy as np
import os
import pickle


evaluator_logger = logger.get_logger('evaluator',
'evaluator' )


def parse_cmdline():
    parser = argparse.ArgumentParser(description='model selector')
    parser.add_argument('X_test',
                        type=str,                    
                        help='test file to train')
    parser.add_argument('y_test',
                        type=str,                    
                        help='y test file')
    parser.add_argument('model_file',
                        type=str,                    
                        help='model_file')


    args = parser.parse_args()
    return args


def evaluator(X, y, model):
    X_test = X.values
    y_test = y.values.reshape(1, -1)[0]
    y_pred = model.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    df = pd.DataFrame({y.columns[0] : y_pred})
    prediction_file = config_parser.configuration.get_file_location('evaluator', 'prediction_file')
    df.to_csv(prediction_file, index=False)
    


def main(args):
    
    df1 = pd.read_csv(args.X_test)
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)
    df2 = pd.read_csv(args.y_test)
    evaluator(df1, df2, model)
    
if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
