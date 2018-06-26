import pandas as pd
import numpy as np
import h5py

def bvReader_h5(filename, dataset = "Data", tsName = "timeSeries", headerOnly = False) :
    """ Read BV format
    """


    with h5py.File(filename, "r") as f:
        label = f["Data"].get("timeSeries").attrs["Header"].astype(str)
        if headerOnly :
            time = []
            data = []
        else :
            time = f["Time"].get("Simulation Time").value
            data = f[dataset].get(tsName).value

    return time, data, label


def bvWriter_h5(filename, xAxis , data, label, datasetName = "Data", tsName = "timeSeries" ):
    """
        Write a TS file in BV format
    """

    with h5py.File(filename, "w") as f:
        timeGoup = f.create_group("Time")
        timeGoup.create_dataset( "Simulation Time", data = xAxis,  dtype='float')
        dataGroup = f.create_group( datasetName )
        dataSet = dataGroup.create_dataset( tsName, data = data,  dtype='float')
        dataSet.attrs["Header"] = ",".join( label )

