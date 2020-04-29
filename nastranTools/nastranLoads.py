from pyNastran.bdf.bdf import BDF
from numpy import zeros, logical_and, where

def getLoadDistribution(fpath, xcut, ilc):
    # read nastran file
    model = BDF()
    model.read_bdf(fpath)
    n = len(xcut)
    fcut = zeros((n,3))
    # get load corresponding with the good id
    if ilc in model.loads.keys():
        load = model.loads[ilc]
        print('Postprocessing LOAD', ilc)
        # loop over all force cards
        for force in load:
            # position of the node, based on the node id in the force card
            x = model.nodes[force.node].xyz[0]
            icut = where(logical_and(x>=xcut[:-1], x<xcut[1:]))
            fcut[icut] += force.xyz
        print('Ok')
    else:
        print('Could not find LOAD', ilc)
    print()
    return fcut
