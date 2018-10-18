import pandas as pd
import numpy as np
import h5py

def bvReader_h5(filename, dataset = "Data", headerOnly = False, usecols=None) :
    """ Read BV format
    """

    with h5py.File(filename, "r") as f:
        ds = f.get(dataset)
        label = ds.dims[1][0].value
        if label.dtype not in [int, float]  :
            label = label.astype(str)
        if headerOnly :
            time = []
            data = []
        else :
            time = ds.dims[0][0].value
            data = ds.value
            
    if usecols is not None:
        useidx = [list(label).index(col) for col in usecols]
        return time, data[:,useidx], label[useidx]
    else:
        return time, data, label


def bvWriter_h5(filename, xAxis , data, labels, datasetName = "Data", compression = None, chunks = None, dtype = "float" ):
    """
        Write a TS file in BV format
    """

    chunksTime = None
    if compression :
        chunks = (len(xAxis),1)

    if chunks is not None :
       chunksTime = (chunks[0], )

    with h5py.File(filename, "w") as f:
        f.create_dataset( "Time", data = xAxis,  dtype=dtype, compression=compression , chunks = chunksTime)
        f.create_dataset( "Channel", data = labels, dtype=h5py.special_dtype(vlen=str), compression=compression)
        f.create_dataset( datasetName, data = data,  dtype=dtype, compression=compression,  chunks=chunks)

        #Set dimension scale
        f[datasetName].dims.create_scale(f["Time"], "Time")
        f[datasetName].dims[0].attach_scale(f["Time"])

        f[datasetName].dims.create_scale(f["Channel"], "Channel")
        f[datasetName].dims[1].attach_scale(f["Channel"])





