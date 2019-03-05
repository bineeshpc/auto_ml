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

feature_selector_logger = logger.get_logger('feature_selector',
'feature_selector' )


def parse_cmdline():
    parser = argparse.ArgumentParser(description='model selector')
    parser.add_argument('X_train',
                        type=str,                    
                        help='X train')
    parser.add_argument('y_train',
                        type=str,                    
                        help='y column value to predict')
    
    args = parser.parse_args()
    return args
        
            
def main(args):
    directory = config_parser.configuration.get_directory('feature_selector')
    
    def copy(src, directory):
        target = os.path.join(directory, os.path.basename(src))
        shutil.copyfile(src, target)

    copy(args.X_train, directory)
    copy(args.y_train, directory)
    

    

if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
