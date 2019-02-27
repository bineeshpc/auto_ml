import yaml
import os
import utils

class ConfigParser:
    def __init__(self):
        stream = open('config.yml', 'r')
        self.configuration = yaml.load(stream)

    def get_directory(self, component):
        final_dir = self.configuration[component]['directory']
        base_dir = self.configuration['output_location']
        directory = os.path.join(base_dir, final_dir)
        utils.generate_directories(directory)
        return directory

    def get_log_location(self, location):
        return os.path.join(self.configuration['output_location'],
        self.configuration[location]['log_location'])

configuration = ConfigParser()
# def generate_files():
#     import os
#     content = "#! /usr/bin/env python\n"
#     for k, v in y.items():
#         with open(v, "w") as f:
#             f.write(content)
#         os.system('chmod u+x {}'.format(v))
        
# generate_files()