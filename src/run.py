from time import time
from diffusion_graph import create_graph
from url_helpers import download_selected_papers

t0 = time()

#create_graph()
download_selected_papers()

print("Total time: %0.3fs." % (time() - t0))
