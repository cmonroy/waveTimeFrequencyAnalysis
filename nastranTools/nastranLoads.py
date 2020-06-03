from pyNastran.bdf.bdf import BDF
from numpy import zeros, logical_and, where

def getLoadDistribution(fpath, xcut):
    # read nastran file
    model = BDF()
    model.read_bdf(fpath)
    n = len(xcut)
    fcut = {}
    # get load corresponding with the good id
    for ilc,load in model.loads.items():
        print('Postprocessing LOAD', ilc)
        # loop over all force cards
        fcut[ilc] = zeros((n,3))
        for force in load:
            # position of the node, based on the node id in the force card
            x = model.nodes[force.node].xyz[0]
            icut = where(logical_and(x>=xcut[:-1], x<xcut[1:]))
            fcut[ilc][icut] += force.xyz
        print('Ok')
    print()
    return fcut
