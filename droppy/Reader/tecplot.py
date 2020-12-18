"""

   Reader for tecplot files

"""

def tecplot_HOS(file):
    """
       Read the 3d.dat from HOS (might be made more general)
    """
    import pandas as pd
    import re

    from io import StringIO


    with open(file, 'r') as a:
        data = a.read()
    blockList = [StringIO(str_) for str_ in data.split("\nZONE")]


    if len( blockList ) > 1 :   # Several "ZONE" block
        print ("Reading surface time series")
        for ibloc in range(1, len(blockList)) :
            #Parse zone information
            blocHeader = blockList[ibloc].readline()
            time = float( re.findall(r"[\S]+", blocHeader.replace(",", " ")  )[2] )
            if ibloc == 1 :
                a = pd.read_csv(  blockList[ibloc] , skiprows = 0, header = None , names = [ "x" , "y" , time ] , usecols = [0,1,2] , delim_whitespace = True, engine = "c" , dtype = float)
            else :
                a[time] = pd.read_csv(  blockList[ibloc] , skiprows = 0, header = None , names = [ time ] , usecols = [0]  , delim_whitespace = True, engine = "c", dtype = float )

        #If 2D :
        a.drop( "y" , axis=1, inplace = True )
        a.set_index( "x" , inplace = True )
        a = a.transpose()
        a.index = a.index.astype(float)
        return a

    else :   # Only one block
        #Parse variables :
        title = blockList[0].readline()
        while title.startswith("#") :
            title = blockList[0].readline()
        var = blockList[0].readline().split("=")[1].split()
        var = [ s[1:-1] for s in var]
        return pd.read_csv(  blockList[0] , skiprows = 0, header = None , names = var, delim_whitespace = True, engine = "c", index_col = 0, dtype = float )



if __name__ == "__main__" :

    print ("Plot tecplot files")

    import argparse
    from droppy.pyplotTools import dfSlider
    parser = argparse.ArgumentParser(description='Visualize HOS 2D results', formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument( '-nShaddow', help='For animation', type = int, default=0)
    parser.add_argument( "inputFile" )
    args = parser.parse_args()

    if args.inputFile[:-4] == ".wif" :
        from Spectral import Wif
        df = Wif( args.inputFile ).Wave2DC( tmin = 0.0 , tmax = 200. , dt = 0.4 , xmin = 0. , xmax = 400. , dx = 4. ,  speed = 0. )
    else :
        df = tecplot_HOS( args.inputFile )
    dfSlider( df )
