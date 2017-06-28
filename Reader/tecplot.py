"""

   Reader for tecplot files

"""

def tecplot_HOS(file):
   """
      Read the 3d.dat from HOS (might be made more general)
   """
   import pandas as pd
   import re
   from cStringIO import StringIO
   arrays = []
   with open(file, 'r') as a:
      data = a.read()
   blockList = [StringIO(str_) for str_ in data.split("\nZONE")]

   if len( blockList ) > 1 :  #Several "ZONE" block

      for ibloc in range(1, len(blockList)) :
         #Parse zone information
         blocHeader = blockList[ibloc].readline()
         time = float( re.findall(r"[\S]+", blocHeader.replace(",", " ")  )[2] )
         if ibloc == 1 :
            a = pd.read_csv(  blockList[ibloc] , skiprows = 0, header = None , names = [ "x" , "y" , time ] , usecols = [0,1,2] , delim_whitespace = True, engine = "c" )
         else :
            a[time] = pd.read_csv(  blockList[ibloc] , skiprows = 0, header = None , names = [ time ] , usecols = [0]  , delim_whitespace = True, engine = "c" )

      #If 2D :
      a.drop( "y" , axis=1, inplace = True )
      a.set_index( "x" , inplace = True )
      a = a.transpose()
      return a

   else :  #Only one block
      #Parse variables :
      while blockList[0].readline().startswith("#") :
         title = blockList[0].readline()
      var = blockList[0].readline().split()[2:]
      var = [ s[1:-1] for s in var]

      #Return dataFrame
      return pd.read_csv(  blockList[0] , skiprows = 0, header = None , names = var, delim_whitespace = True, engine = "c" )


if __name__ == "__main__" :

   import argparse
   from pyplotTools import  dfSlider
   parser = argparse.ArgumentParser(description='Visualize HOS 2D results' ,  formatter_class = argparse.RawTextHelpFormatter)
   parser.add_argument( '-nShaddow',   help='For animation' , type = int,  default=0)
   parser.add_argument( "inputFile" )
   args = parser.parse_args()

   if args.inputFile[:-4] == ".wif" :
      from Spectral import Wif
      df = Wif( args.inputFile ).Wave2DC( tmin = 0.0 , tmax = 200. , dt = 0.4 , xmin = 0. , xmax = 400. , dx = 4. ,  speed = 0. )
   else :
      df = tecplot_HOS( args.inputFile )
   dfSlider( df )