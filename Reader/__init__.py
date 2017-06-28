from __future__ import absolute_import
from Reader import tecplot , bvFormat , openFoam , arianeReader, simpleReader
bvReader = bvFormat.bvReader
bvWriter = bvFormat.bvWriter
tecplot_HOS = tecplot.tecplot_HOS
openFoamReader = openFoam.openFoamReader
ariane8Reader = arianeReader.ariane8Reader
ariane702Reader = arianeReader.ariane702Reader
simpleReader  = simpleReader.simpleReader

#Reader dictionary => possible to pass reader as string
dicoReader = {
              "bvReader" : bvReader ,
              "openFoamReader" : openFoamReader ,
              "ariane702Reader" : ariane702Reader ,
              "simpleReader" : simpleReader ,
              "ariane8Reader" : ariane8Reader ,
              }


dicoWriter = {
             "bvWriter" : bvWriter ,
             }

def dfRead( filename , reader , *args , **kwargs  ) :
   """
      Read and return as a dataFrame
   """
   import pandas as pd
   res = dicoReader[reader] (filename , *args , **kwargs )

   if type(res) == tuple :
      return pd.DataFrame( index = res[0]  , data = res[1] , columns = res[2] )
   else :
      return res
