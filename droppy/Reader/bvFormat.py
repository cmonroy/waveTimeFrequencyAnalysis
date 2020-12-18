import pandas as pd
import numpy as np
import os
import gzip



def openCompressed(filename, mode = "r", compression = "infer" ) :
    if compression == "infer" :
        if os.path.splitext(filename)[-1] == ".gz":
            compression_ = "gzip"
        else :
            compression_ = None
    else :
        compression_ = compression

    if compression_ is None :
        f = open( filename, mode)
    elif compression_ == "gzip" :
        f = gzip.open( filename, mode+"t")

    return f



def bvReader(filename, headerOnly = False, compression = "infer", usecols=None):
    """ Read BV format
    """

    f = openCompressed( filename, "r", compression = compression)

    #Parse header
    line = f.readline()
    while not line.startswith('#UNITS'):
        if line.startswith('#TIME'):
            lab_tmp = line.split()
        line = f.readline()

    labels = lab_tmp[1:]
    if headerOnly : 
       return [] , [] , labels

    if (usecols is not None) and (type(usecols[0]) is not np.int):
        labels_tmp = usecols
        usecols = [0] + [labels.index(u)+1 for u in usecols]
        labels = labels_tmp

    #Fastest option
    df = pd.read_csv(f, comment = "#" , header=None , delim_whitespace=True, dtype = float, usecols=usecols, index_col = 0 )

    if len(labels) != df.shape[1] : labels = [ "Unknown{}".format(j) for j in range(df.shape[1])  ]
    df.columns = labels
    f.close()

    return df


def bvWriter(filename,  xAxis, data , labels=[], units=[], comment='', compression = "infer"):
    """
        Write a TS file in BV format
    """

    rt = '\n'
    try:
        nbTime, nbChannel = np.shape(data)
    except:
        nbTime = np.shape(data)[0]
        nbChannel = 1

    if len(labels) < 1: labels = ['Label-' + str(i+1) for i in range(nbChannel)]
    if len(units) < 1: units = ['Unit-' + str(i+1) for i in range(nbChannel)]


    f = openCompressed( filename, "w", compression = compression)

    if comment.strip() : f.write("# "+comment+rt)
    f.write("#TIMESERIES"+rt)
    f.write("#NBCHANNEL " + str(nbChannel)+rt)
    f.write("#NBTIMESTEPS " + str(nbTime)+rt)
    f.write("#TIME " + " ".join(map(str, labels))+rt)
    f.write("#UNITS " + " ".join(map(str, units))+rt)
    # use numpy method for array dumping and loading
    all = np.empty((nbTime, nbChannel+1), dtype=float)
    all[:,0] = xAxis

    if nbChannel == 1: all[:,1] = data
    else: all[:,1:] = data
    np.savetxt( f, all, fmt = "%.5e" )
    f.close()
    return
