#! /usr/bin/env python

#! /usr/bin/env python

"""

"""

from sklearn.model_selection import train_test_split
import logger
import config_parser
import utils
import argparse
import pandas as pd
import numpy as np
import os
import pickle
import shutil


def parse_cmdline():
    parser = argparse.ArgumentParser(description='model selector')
    parser.add_argument('X_train',
                        type=str,                    
                        help='X train')
    parser.add_argument('y_train',
                        type=str,                    
                        help='y column value to predict')
    parser.add_argument('configfile',
                        type=str,                    
                        help='configfile')
    
    args = parser.parse_args()
    return args
        
            
def main(X_train, y_train, configfile):
    directory = config_parser.get_configuration(configfile).get_directory('feature_selector')
    
    def copy(src, directory):
        target = os.path.join(directory, os.path.basename(src))
        shutil.copyfile(src, target)

    copy(X_train, directory)
    copy(y_train, directory)
    

    

if __name__ == "__main__":
    args = parse_cmdline()
    main(args.X_train, args.y_train, args.configfile)
