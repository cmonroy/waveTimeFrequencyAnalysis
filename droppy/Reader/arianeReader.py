import pandas as pd

def ariane8Reader(filename, labels = None, headerOnly = False, usePandas = True, headerMax = 200) :
   """
      Read ariane8 time series
   """
   with open(filename, 'r') as f :
      i = 0   ; sep=0
      while not sep==3 and  i<headerMax :
         i+=1
         a = f.readline()
         if  a.startswith("_____")  : sep+=1
         if sep == 1 : case1sep = i
   if i == headerMax : i = case1sep  #Some times only one "________" in Ariane files
   
   if headerOnly : 
      return

   if usePandas :
      #Fastest option : pandas (0.3s on test case)
      table = pa.read_csv(filename, skiprows = i, header=None , delim_whitespace=True, dtype = float ).as_matrix()
   else :
      table = np.loadtxt(filename, skiprows = i)
   data = table[:,1:]

   if labels is None :  labels = [ "Unknown{}".format(j) for j in range(len(data[0,:]))  ]
   data.shape = -1 , len(labels)
   return pd.DataFrame(table[:,0], data ,labels)


def ariane702Reader(filename) :
    """
      Read Ariane 7.02 time series
    """
    fin = open(filename, 'r')
    buf = fin.read()
    fin.close()
    lines = buf.split('\n')
    #find the first data line
    for i, line in enumerate(lines):
        if line.startswith('__________________________________________________________________________________'):
            break
    else:
        raise Exception('Separator not found')
    i += 1
    # find the last data line
    lastLine = len(lines)
    while len(lines[lastLine-1]) < 5:
        lastLine = lastLine -1
    # allocate table
    words = lines[i].split()
    nCol = len(words)
    nLin = lastLine - i
    table = np.empty((nLin, nCol), dtype=float)
    # default labels, not read from Ariane File
    labels = [ "Unknown{}".format(j) for j in range(nCol-1)  ]
    # read file content
    for i, line in enumerate(lines[i:]):
        if line:
           table[i, :] = map(float, line.split())

    return pd.DataFrame(table[:,0], table[:,1:],labels)
