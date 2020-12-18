# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:19:53 2020

@author: abenhamou
"""

import os
from os.path import exists
import h5py

def cleanDataset(file,datasets):
    """ Delete dataset(s) from HDF file
    """

    if exists(file):
        with h5py.File(file,  "a") as f:
            if isinstance(datasets, str):
                if datasets in list(f.keys()): del f[datasets]
            else:
                for dset in datasets:
                    if dset in list(f.keys()): del f[dset]
    else:
        print(f'File not found "{file:}"')

if __name__ == "__main__":
    import time
    import pandas as pd
    import numpy as np
    
    hfile = r'C:\Users\abenhamou\Desktop\test_HDF\test.hdf'
    ds = pd.Series(index=np.arange(100),data=150)
    ds.to_hdf(hfile,'ds')
    
    for i in np.arange(1,100):
        df = pd.DataFrame(index=np.arange(1000),columns=np.arange(i),data=np.random.rand(1000,i))
        time.sleep(0.1)
        clean_Dataset(hfile,'test')
        time.sleep(0.1)
        df.to_hdf(hfile,'test')
        print(i, os.path.getsize(hfile))
    hfile2 = r'C:\Users\abenhamou\Desktop\test_HDF\test2.hdf'
    df.to_hdf(hfile2,'test')
    ds.to_hdf(hfile2,'ds')
