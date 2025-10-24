from pprl_toolkit.embedder.embedder import Embedder
from pprl_toolkit.embedder import features
from functools import partial
import pandas as pd
import numpy as np

if __name__  == '__main__': 
    path = "data/test" 
    d1 = pd.read_csv(f"{path}/test1.csv")
    d2 = pd.read_csv(f"{path}/test2.csv")
    factory = dict(
        id = partial(features.gen_misc_features, label="id"),
        name = partial(features.gen_misc_features, label="name"),
        email = partial(features.gen_misc_features, label="email"), 
        phone = partial(features.gen_misc_features, label="phone")
    )
    
    embedder = Embedder(factory, bf_size=1024, num_hashes=2)
    spec1 = dict(id = "id", name = 'name', email = 'email', phone='phone')
    
    edf1 = embedder.embed(d1, colspec=spec1, update_thresholds=True)
    edf2 = embedder.embed(d2, colspec=spec1, update_thresholds=True)    
    X = np.zeros((2, 10))
    X[0,[0,2,4,6,8]] = 1.0
    
    print(X)
        
    similarites = embedder.compare(edf1, edf2)
    print(similarites)
    
    