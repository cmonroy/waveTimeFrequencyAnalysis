import pandas as pd

def bvReader(filename, headerOnly = False ):
    """ Read BV format
    """

    fin = open(filename, 'r')
    buf = fin.read()
    fin.close()
    lines = buf.split('\n')
    #find the header lines
    for i, line in enumerate(lines):
        if line.startswith('#NBCHANNEL'):
            words = line.split()
            nbChannel = int(words[1])
        if line.startswith('#NBTIMESTEPS'):
            words = line.split()
            nbTime = int(words[1])
        if line.startswith('#TIME'):
            lab_tmp = line.split()
        if line.startswith('#UNITS'):
            # not managed yet
            break
    else:
        raise Exception('Could not read header data')

    labels = lab_tmp[1:]
    if headerOnly : 
       return [] , [] , labels


    #Fastest option : pandas (0.3s on test case)
    data = pd.read_csv(filename, comment = "#" , header=None , delim_whitespace=True, dtype = float).as_matrix()
    xAxis = data[:,0]
    data = data[:,1:]
    
    """
    else :
       #print "Pandas not available"
       #Second fastest option (0.7s on test case)
       xAxis = np.zeros( (nbTime) )
       data = np.zeros( (nbTime, nbChannel) )
       iline = 0
       for line in lines :
          if line.strip() and not line.startswith("#") :
             x = map( float, line.split() )  # str => float : 0.3s , line.split() => 0.2s
             xAxis[iline] = x[0]  #0.03s
             data[iline,:] =x[1:] #0.20s
             iline += 1
    #Slowest option : numpy  (1.3s on test case)
    data = np.loadtxt(filename)
    xAxis = data[:,0]
    data = data[:,1:]
    """

    #Check that header are consistent with channels
    if len(labels) != len(data[0,:]) : labels = [ "Unknown{}".format(j) for j in range(len(data[0,:]))  ]

    return pd.DataFrame(index = xAxis , data = data  , columns = labels)


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
