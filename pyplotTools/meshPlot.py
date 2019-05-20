import numpy as np
from matplotlib import pyplot as plt



def plotMesh( coords, quads, tris, ax=None, color = "darkblue" , **kwargs ):
    """Plot a surface mesh
    """

    if ax is None :
        fig, ax = plt.subplots()

    q = np.c_[ (quads[:,:], quads[:,0]) ]
    t = np.c_[ (tris[:,:], tris[:,0]) ]
    ax.plot( coords[ q , 0 ].transpose(), coords[ q , 1 ].transpose() , "-" , color = color , **kwargs)
    ax.plot( coords[ t , 0 ].transpose(), coords[ t , 1 ].transpose() , "-" , color = color , **kwargs)
    return ax


