from time import time
from diffusion_graph import create_graph
from graph_ops import download_papers

t0 = time()

download_papers('cache/top_papers_3.txt', 'out.txt')

print("Total time: %0.3fs." % (time() - t0))
