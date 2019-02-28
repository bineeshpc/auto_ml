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


if  __name__ == "__main__":
    commands = [
        Command('mkdir -p /tmp/test'),
        Command('touch /tmp/test/one'),
        Command('mv /tmp/test/one /tmp/test/two', 'mv /tmp/test/two /tmp/test/one')
    ]

    for cmd in commands:
        cmd.do()
        cmd.undo()
