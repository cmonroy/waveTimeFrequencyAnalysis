
import pandas as pd
from io import StringIO

def read_csv_block( filepath_or_buffer , block_delim = None , startBlock = 0 , endBlock = None,  singleHeader = False , **kwargs     ) :
    """ Read the data by block
    return list of dataframe
    """

    #Split the data
    with open(filepath_or_buffer) as f: data = f.read()
    blockList = [ StringIO(str_) for str_ in data.split(block_delim) ]

    #Remove the end of the line containing the splitter
    for block in blockList : block.readline()

    if endBlock is None : endBlock = len( blockList )

    return [ pd.read_csv( block , **kwargs ) for block in blockList[startBlock : endBlock] ]
