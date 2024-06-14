import os
from pathlib import Path

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


pids = Path("pids.txt").read_text().splitlines()

for pid in pids:
    alive = check_pid(int(pid))
    print(f"{pid}\t{alive}")
