import pandas as pd
import numpy as np

def bvReader(filename, headerOnly = False, readHeader=True, usecols=None):
    """ Read BV format
    """

    if readHeader:
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
    else:
        lab_tmp = []

    labels = lab_tmp[1:]
    if headerOnly : 
       return [] , [] , labels


    #Fastest option : pandas (0.3s on test case)
    data = pd.read_csv(filename, comment = "#" , header=None , delim_whitespace=True, dtype = float, usecols=usecols).as_matrix()
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


def bvWriter(filename,  xAxis, data , labels=None, units=None, comment=''):
    """
        Write a TS file in BV format
    """
    rt = '\n'
    try:
        nbTime, nbChannel = np.shape(data)
    except:
        nbTime = np.shape(data)[0]
        nbChannel = 1
    
    if labels==None: labels = ['Label-'+str(i+1) for i in range(nbChannel)]
    if units==None: units = ['Unit-'+str(i+1) for i in range(nbChannel)]
    
    
    f = open(filename, 'w')
    f.write("# "+comment+rt)
    f.write("#TIMESERIES"+rt)
    f.write("#NBCHANNEL " + str(nbChannel)+rt)
    f.write("#NBTIMESTEPS " + str(nbTime)+rt)
    f.write("#TIME " + " ".join(map(str, labels))+rt)
    f.write("#UNITS " + " ".join(map(str, units))+rt)
    # use numpy method for array dumping and loading
    all = np.empty((nbTime, nbChannel+1), dtype=float)
    all[:,0] = xAxis
    if nbChannel==1: all[:,1] = data
    else: all[:,1:] = data
    np.savetxt( f, all )
    f.close()
    return
