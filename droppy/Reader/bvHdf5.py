import pandas as pd
import numpy as np
import h5py
import os

def bvReader_h5(filename, dataset = "Data", headerOnly = False, usecols=None, swmr=True, swmr_trunc=None):
    """ Read BV HDF5 format

    Parameters
    ----------
    filename : str
        File name.
    dataset : str, optional
        Dataset identifier. The default is "Data".
    headerOnly : bool, optional
        Read only header. The default is False.
    usecols : array-like, optional
        List of columns labels to read. If None, all columns are read. The default is None.
    swmr : bool, optional
        Read file with SWMR function. The default is True.
    swmr_trunc : bool, optional
        Truncate the tail of the signal if only zeros are encountered. The truncation is based on time axis. The default is True if swmr, False if not swmr.

    Returns
    -------
    time : numpy array
        Array of time steps.
    data : numpy array
        Data array.
    label : numpy array
        Columns labels.
    """

    if not os.path.exists(filename) :
        raise(OSError (2 ,'No such file or directory', filename ))

    with h5py.File(filename, "r", swmr=swmr) as f:
        ds = f.get(dataset)
        label = ds.dims[1][0][()]
        if label.dtype not in [int, float]:
            label = label.astype(str)
        if headerOnly :
            time = []
            data = []
        else :
            time = ds.dims[0][0][()]
            data = ds[()]


    if (( swmr and (swmr_trunc is not False)) or (swmr_trunc is True)) and not headerOnly:
        idx = max(np.nonzero(time)[0])  # Find maximum non zero index
        time = time[:idx+1]
        data = data[:idx+1,:]

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
        f["Time"].make_scale("Time")
        f[datasetName].dims[0].attach_scale(f["Time"])

        f["Channel"].make_scale("Channel")
        f[datasetName].dims[1].attach_scale(f["Channel"])





