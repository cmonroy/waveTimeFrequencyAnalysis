###########################################################################
# This Python file is used to read any input file and store in an object. #
# The following must be provided to call this function :                  #
#   >> a list of input keywords sorted by type                            #
#   >> a dictionary with default parameters (optional)                    #
###########################################################################
from __future__ import print_function
import os, sys, configparser, ast
import numpy as np

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
def getBool(string):
    if string in ['True','true','T','t','1']:
        return True
    elif string in ['False','false','F','f','0']:
        return False
    else:
        print('Invalid boolean entry : '+str(string))
        raise SystemExit('')
        
#Use this function to read an input file (.ini type) and store in dictionary
# --> iParam : list of integer parameters to read (list of string)
# --> rParam : list of real parameters to read (list of string)
# --> lParam : list of logical parameters to read (list of string)
# --> sParam : list of string parameters to read (list of string)
# --> aParam : list of array parameters to read (list of string)
# --> dParam : list of dictionary parameters to read (list of string)
# --> default: dictionary of default parameters
# --> name : name of section to read

def readInput(filename, iParam=[], rParam=[], lParam=[], sParam=[], aParam=[], dParam=[], default={}, name = 'input'):
    if not (os.path.isfile(filename)):
        print('ERROR: Cannot find input file "'+filename+'"')
        os._exit(1)
    
    params = dict(default)
    config = configparser.ConfigParser()
    config.read(filename)
       
    print("Read input parameters from file:", filename)
    
    # integer parameters
    for ip in iParam:
        try:
            txt = str(config[name][ip])
            params[ip] = int(txt)
        except KeyError: pass
    
    # real parameters
    for rp in rParam:
        try:
            txt = str(config[name][rp])
            params[rp] = float(txt)
        except KeyError: pass
    
    # logical parameters
    for lp in lParam:
        try:
            txt = str(config[name][lp])
            params[lp] = getBool(txt)
        except KeyError: pass
        
    # string parameters
    for sp in sParam:
        try:
            txt = str(config[name][sp])
            txt = txt.replace("'","")
            txt = txt.replace('"','')
            params[sp] = txt
        except KeyError: pass
        
    # array parameters
    for ap in aParam:
        try:
            txt = str(config[name][ap])
            params[ap] = np.array(ast.literal_eval(txt))
        except KeyError: pass
        
    # dictionary parameters
    for dp in dParam:
        try:
            txt = str(config[name][dp])
            params[dp] = dict(ast.literal_eval(txt))
        except KeyError: pass
    
    spar = Struct(**params)
    return spar  