#! /usr/bin/env python

"""
Given a data frame split the data based on data type
Known data types are

Numerical
Categorical
Text

"""
import logger
import argparse
import os
import config_parser


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


def main(args):
    cnf = config_parser.get_configuration(args.configfile)
    input_location = cnf.configuration['input_location']
    transformer_prg = os.path.join(input_location, 'transformer.py')

    cmd = "{transformer_prg} {bool_file} {categorical_file} {text_file} {numerical_file} {configfile}"
    cmd = cmd.format(transformer_prg=transformer_prg,
                     bool_file=args.bool_,
                     categorical_file=args.categorical,
                     text_file=args.text,
                     numerical_file=args.numerical,
                     configfile=args.configfile
    )
    os.system(cmd)


if __name__ == "__main__":
    args = parse_cmdline()
    transformers_logger = logger.get_logger('transformer',
                                            'transformer',
                                            args.configfile)
    main(args)
