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
                        help='Configfile')

    args = parser.parse_args()
    return args

def write_csv(all_types, filename):
    def get_total_lines(file_):
        count = 0
        with open(file_) as f:
            for line in f:
                count += 1
        return count

    all_types_list = list(all_types)
  
    line_nums = []
    for type_, file_ in all_types_list:
        line_num = get_total_lines(file_)
        line_nums.append(line_num)

    # assume that line numbers can be 0 and all other entries will be same
    unique_nums = len(set(line_nums))
    assert ((unique_nums == 2) or (unique_nums == len(line_nums)))
    
    relevant_nums = [index for index, line_num in enumerate(line_nums) if line_num > 1]
    assert len(set([line_nums[i] for i in relevant_nums])) == 1


    relevant_types = [all_types_list[i] for i in relevant_nums]

    open_files = [open(file_) for _, file_ in relevant_types]

    def get_one_line(file_):
        return file_.readline().strip('\n')

    def get_all_lines(open_files):
        return [get_one_line(file_) for file_ in open_files]

    num_lines = max(line_nums)
    with open(filename, 'w') as f:
        for i in range(num_lines):
            f.write('{}\n'.format(','.join(get_all_lines(open_files))))
    
    
def main(args):            
    bool_ = args.bool_
    categorical = args.categorical
    text = args.text
    numerical = args.numerical

    type_s = ['bool_', 'categorical', 'text', 'numerical']
    file_s = [bool_, categorical, text, numerical]

    all_types = zip(type_s, file_s)
    
    directory = config_parser.get_configuration(args.configfile).get_directory('stacker')
    filename = '{}.csv'.format('X_train')
    filename = os.path.join(directory, filename)

    write_csv(all_types, filename)


        

if __name__ == "__main__":
    args = parse_cmdline()
    
    transformers_logger = logger.get_logger('transformer',
                                            'transformer',
                                            args.configfile)

    main(args)
