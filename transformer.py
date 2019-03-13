#! /usr/bin/env python

"""
Given a data frame split the data based on data type
Known data types are

Numerical
Categorical
Text

"""
import logger
import config_parser
import utils
import argparse
import pandas as pd
import numpy as np
import os
from exploratory_data_analysis import detect_outliers
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler



def parse_cmdline():
    parser = argparse.ArgumentParser(description='Input directory')
    parser.add_argument('bool_',
                        type=str,                    
                        help='bool type data')
    parser.add_argument('categorical',
                        type=str,                    
                        help='categorical type data')
    parser.add_argument('text',
                        type=str,                    
                        help='text type data')
    parser.add_argument('numerical',
                        type=str,                    
                        help='numerical type data')
    parser.add_argument('configfile',
                        type=str,                    
                        help='configfile')

    args = parser.parse_args()
    return args


class Transformer:
    def __init__(self):
        """ Transformer is a 4 tuple
        tuple values can be empty
        
        """
        self.transformer = None
        try:
            self.add_logger()
        except:
            self.logger = logger.get_logger2('transformer',
                                            '/tmp/transformer.log')


    def add_logger(self, logger):
        self.logger = logger

    def do_transformation(self, name, function, args, kwargs):
        """ Add a transformer 
        (a string representing name of transformer function,
        transformer function,
        args is a tuple or list that can be used a positional function arguments,
        kwargs is a dictionary for key word functional arguments)
        """
        self.logger.debug("Adding {} {} {} {}".format(name, function, args, kwargs))
        self.logger.info('Adding {} {}'.format(name, function))
        self.transformer = (name, function, args, kwargs)
        return self.execute(function, args, kwargs)
        
    def execute(self,function, args, kwargs):
        self.logger.debug("Executing {} {} {}".format(function, args, kwargs))
        return function(*args, **kwargs)


def main(args):
    def read_csv(location):
        try:
            df = pd.read_csv(location)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return df
            
    pickle_filename = '/tmp/arguments.pkl' 
    utils.dump(args, pickle_filename)
    bool_ = read_csv(args.bool_)
    categorical = read_csv(args.categorical)
    text = read_csv(args.text)
    numerical = read_csv(args.numerical)

    type_s = ['bool_', 'categorical', 'text', 'numerical']
    dfs = [bool_, categorical, text, numerical]

    all_types = dict(zip(type_s, dfs))
    
    if args.configfile != 'predictor.yml': # dirty hack, need not drop for prediction
        outliers_indices = detect_outliers(numerical, ["Age","SibSp","Parch","Fare"], n=2) # remove if more than 2 outliers

    import titanic 
    # imported here to avoid initialization problems
    # especially pickle_filename should be initialized before importing
    transform_bool = titanic.transform_bool
    transform_categorical = titanic.transform_categorical
    transform_text = titanic.transform_text
    transform_numerical = titanic.transform_numerical

    for type_, df in all_types.items():
        transformer = Transformer()
        directory = config_parser.get_configuration(args.configfile).get_directory('transformer')
        transformers_logger.info(directory)

        filename = '{}.csv'.format(type_)
        filename = os.path.join(directory, filename)
        transformers_logger.info(type_)
    
        if args.configfile != 'predictor.yml': # dirty hack, need not drop for prediction
            df = transformer.do_transformation('remove outliers', titanic.drop_outliers, (df, outliers_indices), {})

        df = transform_bool(type_, df)
        df = transform_categorical(type_, df)
        df = transform_text(type_, df)
        df = transform_numerical(type_, df)

        df.to_csv(filename, index=False)


if __name__ == "__main__":
    args = parse_cmdline()
    transformers_logger = logger.get_logger('transformer',
                                            'transformer',
                                            args.configfile)
    main(args)
