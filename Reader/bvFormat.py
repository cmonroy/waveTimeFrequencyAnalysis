import pandas as pd
import numpy as np

def bvReader(filename, headerOnly = False, readHeader=True, usecols=None):
    """ Read BV format
    usecols : columns to read (position or columns name ( for instance ["TIME" , "Roll" ]  )
    headerOnly : do not read the data, only return header
    """

    if readHeader:
        with open(filename, "r") as f :
            #find the header lines
            for i in range(8) :
                line = f.readline()
                if line.startswith('#NBCHANNEL'):
                    words = line.split()
                    nbChannel = int(words[1])
                if line.startswith('#NBTIMESTEPS'):
                    words = line.split()
                    nbTime = int(words[1])
                if line.startswith('#TIME'):
                    labels = line.split()
                    labels[0] = labels[0][1:]  # "#TIME" => "TIME"
                if line.startswith('#UNITS'):
                    # not managed yet
                    break
            else:
                raise Exception('Could not read header data')

        if headerOnly :
            return [] , [] , labels
    else:
        labels = None

    #Fastest option : pandas (np.loadtxt is much slower)
    data = pd.read_csv(filename, comment = "#" , header = None, names = labels, delim_whitespace=True, dtype = float, usecols = usecols, index_col = 0 )

    #Check that header are consistent with channels
    if labels is None :
        data.columns = [ "Unknown_{}".format(j) for j in range(data.shape[1]) ]

    return data


def bvWriter(filename,  xAxis, data , labels):
    """
        Write a TS file in BV format
    """
    nbTime, nbChannel = np.shape(data)
    fichier = open(filename, 'w')
    print >> fichier, "#"
    print >> fichier, "#TIMESERIES"
    print >> fichier, "#NBCHANNEL " + str(nbChannel)
    print >> fichier, "#NBTIMESTEPS " + str(nbTime)
    print >> fichier, "#TIME " + " ".join(map(str, labels))
    print >> fichier, "#UNITS" + nbChannel*" Unk.unit"
    # use numpy method for array dumping and loading
    all = np.empty((nbTime, nbChannel+1), dtype=float)
    all[:,0] = xAxis
    all[:,1:] = data
    np.savetxt( fichier, all )
    fichier.close()
    return


