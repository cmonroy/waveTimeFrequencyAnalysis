import os
import pandas as pd
import numpy as np

class section(object):

    def __init__(self):
        self.RP = None
        self.COG = None
        self.VIS44PC = None
        self.matrix = None

class donFile(object):

    def __init__(self):
        self.mass = None
        self.COG = None
        self.GYR = None
        self.sections = []

    # read don file
    def read(self,filename):

        nsect = 0
        f = open(filename,'r')
        itf = iter(f)
        for line in itf:
            if 'MASS_BODY' in line:
                self.mass = float(line.split()[-1])
                
            if 'COGPOINT_BODY' in line:
                self.COG = [float(i) for i in line.split()[-3:]]
                
            if 'GYRADIUS_BODY' in line:
                self.GYR = [float(i) for i in line.split()[-6:]]
            
            if 'SECTION_No' in line:
                nsect += 1
                isect = float(line.split()[-1])
                if isect!=nsect:
                    print('ERROR: Problem occurred while reading sections !')
                    os._exit()
                sect = section()
                pline = next(itf)
                if 'REFPOINT' in pline:
                    sect.RP = [float(i) for i in pline.split()[-3:]]
                else:
                    print('ERROR: Problem occurred while reading section REFPOINT !')
                    os._exit()
                pline = next(itf)
                if 'COGFROMAP' in pline:
                    sect.COG = [float(i) for i in pline.split()[-3:]]
                else:
                    print('ERROR: Problem occurred while reading section COGFROMAP !')
                    os._exit()
                pline = next(itf)
                if 'VIS44PC' in pline:
                    sect.VIS44PC = float(pline.split()[-1])
                else:
                    print('ERROR: Problem occurred while reading section VIS44PC !')
                    os._exit()
                pline = next(itf)
                if 'INERTMATRIX' in pline:
                    pline = next(itf)
                    sect.matrix = np.array([float(i) for i in pline.split()])
                    pline = next(itf)
                    while not 'ENDSECTION' in pline:
                        sect.matrix = np.vstack([sect.matrix,[float(i) for i in pline.split()]])
                        pline = next(itf)
                    if 'ENDSECTION' in pline:
                        self.sections.append(sect)
                    else:
                        print('ERROR: Problem occurred while reading ENDSECTION !')
                        os._exit()
                else:
                    print('ERROR: Problem occurred while reading section INERTMATRIX !')
                    os._exit()
        f.close()

    # read don file
    def write(self,filename,info=''):

        f = open(filename,'w')
        f.write('# Project : .don file written using donFormat.py\n')
        f.write('# Info : '+info+'\n')
        f.write('#\n\n')
        f.write('MASS_BODY   1       {:12.5E}\n\n'.format(self.mass))
        f.write('COGPOINT_BODY   1'+(3*' {:8.3f}').format(*self.COG)+'\n\n')
        f.write('GYRADIUS_BODY   1'+(6*' {:8.3f}').format(*self.GYR)+'\n\n')
        
        nSect = len(self.sections)
        for iSect in range(nSect):
            f.write('SECTION_No {:3d}\n'.format(iSect+1))
            f.write('  REFPOINT   '+(3*' {:8.3f}').format(*self.sections[iSect].RP)+'\n')
            f.write('  COGFROMAP  '+(3*' {:8.3f}').format(*self.sections[iSect].COG)+'\n')
            f.write('  VIS44PC   {:12.5E}\n'.format(self.sections[iSect].VIS44PC))
            f.write('  INERTMATRIX\n')
            nlin = self.sections[iSect].matrix.shape[0]
            ncol = self.sections[iSect].matrix.shape[1]
            for i in range(nlin):
                f.write((ncol*' {:12.5E}').format(*self.sections[iSect].matrix[i,:])+'\n')
            f.write('ENDSECTION\n\n')
            
        f.write('ENDFILE\n')
        f.close()
        
    #change section reference point from R0 to R1
    def changeRP(self,iSect,R0,R1):
        CG = self.sections[iSect].COG
        M = self.sections[iSect].matrix[0,0]
        #Move inertia terms
        self.sections[iSect].matrix[3,3] = self.sections[iSect].matrix[3,3] + M*( 2*CG[2]*(R0[2]-R1[2]) + 2*CG[1]*(R0[1]-R1[1]) + R1[2]**2 - R0[2]**2 + R1[1]**2 - R0[1]**2 )
        self.sections[iSect].matrix[4,4] = self.sections[iSect].matrix[4,4] + M*( 2*CG[2]*(R0[2]-R1[2]) + 2*CG[0]*(R0[0]-R1[0]) + R1[2]**2 - R0[2]**2 + R1[0]**2 - R0[0]**2 )
        self.sections[iSect].matrix[5,5] = self.sections[iSect].matrix[5,5] + M*( 2*CG[0]*(R0[0]-R1[0]) + 2*CG[1]*(R0[1]-R1[1]) + R1[0]**2 - R0[0]**2 + R1[1]**2 - R0[1]**2 )
        self.sections[iSect].matrix[3,4] = self.sections[iSect].matrix[3,4] + M*( (CG[0]-R0[0])*(CG[1]-R0[1]) - (CG[0]-R1[0])*(CG[1]-R1[1]) )
        self.sections[iSect].matrix[4,3] = self.sections[iSect].matrix[3,4]
        self.sections[iSect].matrix[3,5] = self.sections[iSect].matrix[3,5] + M*( (CG[0]-R0[0])*(CG[2]-R0[2]) - (CG[0]-R1[0])*(CG[2]-R1[2]) )
        self.sections[iSect].matrix[5,3] = self.sections[iSect].matrix[3,5]
        self.sections[iSect].matrix[4,5] = self.sections[iSect].matrix[4,5] + M*( (CG[1]-R0[1])*(CG[2]-R0[2]) - (CG[1]-R1[1])*(CG[2]-R1[2]) )
        self.sections[iSect].matrix[5,4] = self.sections[iSect].matrix[4,5]
        
        #Move off-diagonal terms
        XGC = CG[0] - R1[0]
        YGC = CG[1] - R1[1]
        ZGC = CG[2] - R1[2]
        self.sections[iSect].matrix[0,4] = +1.*M*ZGC
        self.sections[iSect].matrix[0,5] = -1.*M*YGC
        self.sections[iSect].matrix[1,3] = -1.*M*ZGC
        self.sections[iSect].matrix[1,5] = +1.*M*XGC
        self.sections[iSect].matrix[2,3] = +1.*M*YGC
        self.sections[iSect].matrix[2,4] = -1.*M*XGC
        
        self.sections[iSect].matrix[3,1] = -1.*M*ZGC
        self.sections[iSect].matrix[3,2] = +1.*M*YGC
        self.sections[iSect].matrix[4,0] = +1.*M*ZGC
        self.sections[iSect].matrix[4,2] = -1.*M*XGC
        self.sections[iSect].matrix[5,0] = -1.*M*YGC
        self.sections[iSect].matrix[5,1] = +1.*M*XGC
        
        #Reset reference
        self.sections[iSect].RP = R1