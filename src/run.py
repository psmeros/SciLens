from time import time
from pathlib import Path

#Scilens Directory
scilens_dir = str(Path.home()) + '/Dropbox/scilens/'


t0 = time()

print("Total time: %0.3fs." % (time() - t0))
