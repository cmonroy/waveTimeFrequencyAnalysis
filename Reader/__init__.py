from __future__ import absolute_import
import os
from . import tecplot , bvFormat , openFoam , arianeReader, simpleReader, bvHdf5
bvReader = bvFormat.bvReader
bvWriter = bvFormat.bvWriter
tecplot_HOS = tecplot.tecplot_HOS
openFoamReader = openFoam.openFoamReader
ariane8Reader = arianeReader.ariane8Reader
ariane702Reader = arianeReader.ariane702Reader
simpleReader  = simpleReader.simpleReader
bvReader_h5 = bvHdf5.bvReader_h5
bvWriter_h5 = bvHdf5.bvWriter_h5


#Reader dictionary => possible to pass reader as string
dicoReader = {
              "bvReader" : bvReader ,
              "openFoamReader" : openFoamReader ,
              "ariane702Reader" : ariane702Reader ,
              "simpleReader" : simpleReader ,
              "ariane8Reader" : ariane8Reader ,
              "bvReader_h5" : bvReader_h5 ,
              }

dicoWriter = {
             "bvWriter" : bvWriter ,
             "bvWriter_h5" : bvWriter_h5 ,
             }

def dfRead( filename , reader = "auto", **kwargs  ) :
    """
       Read and return as a dataFrame
    """
    import pandas as pd

    if reader == "auto" :
        """
        Choose reader based on extension
        """
        if os.path.splitext(filename)[-1] == ".ts" : reader = "bvReader"
        elif os.path.splitext(filename)[-1] == ".h5" : reader = "bvReader_h5"
        else : raise(Exception("Can not infer reader type for " + filename))

    if reader not in dicoReader.keys() :
        print ("Unknown reader, please choose within : {}".format(  list(dicoReader.keys() ) ))
        return

    res = dicoReader[reader] (filename , **kwargs )

    if type(res) == tuple :
        return pd.DataFrame( index = res[0]  , data = res[1] , columns = res[2] )
    else :
        return res
