import subprocess, os
def gitdescribe(f):
    return subprocess.run("git describe --always".split(),
                          cwd=os.path.dirname(f) or '.',
                          stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

__version__ = gitdescribe(__file__)
