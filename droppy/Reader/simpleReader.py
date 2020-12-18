import numpy as np
import pandas as pd

def simpleReader(filename, labels = False, headerOnly = False):
   """
   Simplest file format
   #Time label1
   t1  v1
   t2  v2
   ...
   """
   

   if headerOnly :  #Then just count the number of channels
      #TODO
      pass


   all = np.loadtxt(filename)
   xAxis = all[:,0]
   data = all[:,1:]
   ns = len(data[0,:])

   labelsOk = False
   if labels :
      if len(labels) == ns : labelsOk = True

   #Try to read a header with label information
   if not labelsOk :
      try :
         with open(filename, 'r') as fin :
            line = fin.readline().strip()
         if line[0] == "#" :
            labels = line.strip().split()[1:]
            if len(labels) == ns :
               labelsOk = True
         else :
            labelsOk = False
      except :
         labelsOk = False

   #Default labels
   if not labelsOk :
      labels = [ "Unknown{}".format(j) for j in range(len(data[0,:]))  ]

   #return xAxis , data  , labels
   return pd.DataFrame(index=xAxis , data=data , columns=labels)
