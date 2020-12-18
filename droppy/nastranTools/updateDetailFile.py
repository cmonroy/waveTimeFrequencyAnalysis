import os
import sys
from pyNastran.bdf.bdf import BDF
from tkinter import filedialog

def get_new_ids(ids, model, map_n2e):
    new_ids = []
    for eid in ids:
        new_ids.append(eid)
        nodes = model.elements[eid].nodes
        for nid in nodes:
            new_ids += map_n2e[nid]
    return new_ids

def updateDetailFile(datFile, txtFile, n):

    # read fine mesh
    model = BDF()
    model.read_bdf(datFile)
    map_n2e = model.get_node_id_to_element_ids_map()

    # get elt ids defined in the txt file
    print('Updating ', txtFile)
    f = open(txtFile, 'r')
    data = f.read()
    i1 = data.index('SET')+6
    i2 = data.index('ENDSET')-1
    ids = map(int,data[i1:i2].split('\n'))

    # get new ids + remove duplicate
    # do it more than once if you want to extend the area where you want the RAOs
    for i in range(n):
        ids = list(set(get_new_ids(ids, model, map_n2e)))

    # write the new txt file
    f = open(txtFile.replace('.txt', '_ext.txt'), 'w')
    f.write(data[:i1]+'\n'.join(map(str,ids))+data[i2:])
    f.close()
    print('ok')

if __name__ == "__main__":

    if '-n' in sys.argv:
        n = int(sys.argv[sys.argv.index('-n')+1])
    else:
        n = 1
    txtFile = filedialog.askopenfilename(
        initialdir = '.', 
        title = 'Select Homer file with element set (keyword SET/ENDSET)', 
        filetypes = (('Homer set files','*.txt'),) )
    datFile = filedialog.askopenfilename(
        initialdir = os.path.dirname(txtFile), 
        title = 'Select corresponding Nastran file', 
        filetypes = (('Nastran files','*.dat'),) )
    updateDetailFile(datFile, txtFile, n)
