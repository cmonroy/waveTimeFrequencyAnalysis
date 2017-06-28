import numpy as np
import pandas as pd



def openFoamReader(filename, *args , **kwargs) :
    import os
    dicoReader = {
                 "motions.dat" : OpenFoamReadMotion,
                 "sixDofDomainBody.dat" : OpenFoamReadMotion,
                 "surfaceElevation.dat" : OpenFoamReadMotion,
                 "fx.dat" : OpenFoamReadMotion,
                 "fy.dat" : OpenFoamReadMotion,
                 "fz.dat" : OpenFoamReadMotion,
                 "mx.dat" : OpenFoamReadMotion,
                 "my.dat" : OpenFoamReadMotion,
                 "mz.dat" : OpenFoamReadMotion,
                 "PTS_localMotion_pos.dat" : OpenFoamReadMotion,
                 "PTS_localMotion_vel.dat" : OpenFoamReadMotion,
                 "PTS_localMotion_acc.dat" : OpenFoamReadMotion,
                 "forces.dat" : OpenFoamReadForce,
                 "fFluid.dat" : OpenFoamReadForce,
                 "mFluid.dat" : OpenFoamReadForce,
                 "fCstr.dat" : OpenFoamReadForce,
                 "mCstr.dat" : OpenFoamReadForce,
                 "acc.dat" : OpenFoamReadForce
                 }
    fname = os.path.basename(filename) 
    return dicoReader.get(fname , OpenFoamReadMotion ) ( filename , *args , **kwargs )
    # .get(fname, OpenFoamReadMotion) means that the method OpenFoamReadMotion is used by default (if filename is not in dicoReader)

def OpenFoamReadForce(fileName, field = "total"):
   """
   Read openFoam "forces" file
   """
   fil = open(filename, "r")
   data = [ l.strip().strip().replace("(", " ").replace(")", " ").split() for l in fil.readlines() if not l.startswith("#") ]
   fil.close()
   xAxis = np.array(  [float(l[0]) for l in data] )
   nx = len(xAxis)
   ns = len(data[0]) -1
   parsedArray = np.zeros( (nx,ns)  )
   if field == "total" or field == "pressure" :
      dataArray = np.zeros( (nx,6)  )
      labels = ["Fx" , "Fy" , "Fz" , "Mx" , "My" , "Mz"]

   for i, l in enumerate(data) :
      parsedArray[i,:] = map(float , l[1:])

   if ns == 12 :
      if field == "total"  :
         for i in range(3) :
            dataArray[:,i]   = parsedArray[:,0+i] + parsedArray[:,3+i]
            dataArray[:,i+3] = parsedArray[:,6+i] + parsedArray[:,9+i]
      else :
         dataArray = parsedArray
         labels = ["Fx-Pressure" , "Fy-Pressure" , "Fz-Pressure",
                   "Fx-Viscous" , "Fy-Viscous" , "Fz-Viscous",
                   "Mx-Pressure" , "My-Pressure" , "Mz-Pressure",
                   "Mx-Viscous" , "My-Viscous" , "Mz-Viscous", ]
   elif ns == 18 :
      if field == "total" :
         for i in range(3) :
            dataArray[:,i]   = parsedArray[:,0+i] + parsedArray[:,3+i] + parsedArray[:,6+i]
            dataArray[:,i+3] = parsedArray[:,9+i] + parsedArray[:,12+i] + parsedArray[:,15+i]
      elif field == "pressure" :
         for i in range(3) :
            dataArray[:,i]   = parsedArray[:,0+i]
            dataArray[:,i+3] = parsedArray[:,9+i]

      else :
         dataArray = parsedArray
         labels = ["Fx-Pressure" , "Fy-Pressure" , "Fz-Pressure",
                   "Fx-Viscous" , "Fy-Viscous" , "Fz-Viscous",
                   "Fx-Porous" , "Fy-Porous" , "Fz-Porous",
                   "Mx-Pressure" , "My-Pressure" , "Mz-Pressure",
                   "Mx-Viscous" , "My-Viscous" , "Mz-Viscous",
                   "Mx-Porous" , "My-Porous" , "Mz-Porous",
                   ]

   else :
      dataArray = parsedArray
      labels = [ "Unknown{}".format(j) for j in range(ns)  ]
   return  pd.DataFrame(index=xAxis , data=dataArray , columns=labels)

def OpenFoamReadMotion( filename , namesLine = 1 ) :
   """
   Read motion and internal loads from foamStar
   """

   #Read header
   with open(filename, "r") as fil :
      header = [ l.strip().split() for l in [ fil.readline() for line in range(namesLine+1) ]  if l.startswith("#") ]
   print header
   df = pd.read_csv( filename , comment = "#" , header = None ,  delim_whitespace=True, dtype = float , index_col = 0 , names = header[namesLine][2:])
   return df






 

