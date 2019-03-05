#! /usr/bin/env python

import subprocess 

class Command:
    def __init__(self, cmd, undo_cmd=None):
        self.cmd = cmd
        self.undo_cmd = undo_cmd
        
    def do(self):
        print('running {}'.format(self.cmd))
        self.output = subprocess.run(self.cmd.split())

    def undo(self):
        if self.undo_cmd is None:
            self.undo_output = None
        else:
            print('running {}'.format(self.undo_cmd))
            self.undo_output = subprocess.run(self.undo_cmd.split())

    def __repr__(self):
        return "cmd is \n{}\nundo_cmd is\n{}\n-----".format(self.cmd, self.undo_cmd)


class JobRunner:
    def __init__(self):
        self.commands = []

    def add_command(self, cmd_string):
        self.commands.append(Command(cmd_string))

    def execute(self, stop_on_error=False):
        for cmd in self.commands:
            try:
                cmd.do()
                if cmd.output.returncode != 0:
                    break
            except:
                break
        
        

if  __name__ == "__main__":
    commands = [
        Command('mkdir -p /tmp/test'),
        Command('touch /tmp/test/one'),
        Command('mv /tmp/test/one /tmp/test/two', 'mv /tmp/test/two /tmp/test/one')
    ]

    for cmd in commands:
        cmd.do()
        cmd.undo()

    jr = JobRunner()
    jr.add_commamnd('ls')
    jr.add_commamnd("badcmd")
    jr.add_commamnd('pwd')
    jr.execute()
