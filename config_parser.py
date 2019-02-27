import yaml

stream = open('config.yml', 'r')
configuration = yaml.load(stream)

def generate_files():
    import os
    content = "#! /usr/bin/env python\n"
    for k, v in y.items():
        with open(v, "w") as f:
            f.write(content)
        os.system('chmod u+x {}'.format(v))
        
# generate_files()