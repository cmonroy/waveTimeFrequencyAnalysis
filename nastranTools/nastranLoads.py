from pyNastran.bdf.bdf import BDF
from numpy import zeros, logical_and, where

def getLoadDistribution(fpath, xcut, ilc):
    # read nastran file
    model = BDF()
    model.read_bdf(fpath)
    n = len(xcut)
    fcut = zeros((n,3))
    # loop over all load cases written in the analysis
    # only the one with loadid=ilc is post-treated
    for loadid, load in model.loads.items():
        if loadid==ilc:
            print('Load:', loadid)
            # loop over all force cards
            for force in load:
                # position of the node, based on the node id in the force card
                x = model.nodes[force.node].xyz[0]
                icut = where(logical_and(x>=xcut[:-1], x<xcut[1:]))
                fcut[icut] += force.xyz
    return fcut
