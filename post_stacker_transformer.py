#! /usr/bin/env python

import argparse
import os
import config_parser

def parse_cmdline():
    parser = argparse.ArgumentParser(description='post stacker transformer')
    parser.add_argument('X_train',
                        type=str,                    
                        help='X train')

    parser.add_argument('configfile',
                        type=str,                    
                        help='configfile')
    args = parser.parse_args()
    return args

def main(args):
    cnf = config_parser.get_configuration(args.configfile)
    input_location = cnf.configuration['input_location']
    transformer_prg = os.path.join(input_location, 'post_stacker_transformer.py')
    cmd = "{transformer_prg} {X_train} {configfile}"
    cmd = cmd.format(transformer_prg=transformer_prg,
                     X_train=args.X_train,
                     configfile=args.configfile
    )
    os.system(cmd)
   
if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
