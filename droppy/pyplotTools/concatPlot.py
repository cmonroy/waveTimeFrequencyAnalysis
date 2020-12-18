# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:29:38 2020

@author: abenhamou
"""

import os
from os.path import join
import numpy as np


def readImage(path):
    #Import here to avoid mandatory dependence to PIL droppy
    from PIL import Image, ImageDraw, ImageFont
    
    return Image.open(path)

def concatPlot(ims,nline=1,ncol=1,offset=[0,0],title=None,fontsize=30):


    """
    Concatenate a list of images

    Parameters
    ----------
    ims : list of Pillow.Image
        List of images to be concatenated.
    nline : int, optional
        Number of lines. The default is 1.
    ncol : int, optional
        Number of columns. The default is 1.
    offset : int 1D-array, optional
        Horizontal and vertical added spacing between images. The default is [0,0].
    title : str, optional
        Text to be added as title. The default is None.
    fontsize : int, optional
        Font size. The default is 30.

    Returns
    -------
    dst : Pillow.Image
        Returns the concatenated image. It can be saved by using Pillow.Image.save(...).
    """

    #Import here to avoid mandatory dependence to PIL droppy
    from PIL import Image, ImageDraw, ImageFont

    #Check number of lineas and columns
    if nline*ncol <len(ims):
        print('WARNING: number of slots ({}) is lower than number of images ({}).'.format(nline*ncol,len(ims)))
    
    #Compute new image dimensions
    split_idx = np.append(np.arange(len(ims)),np.repeat(np.nan,nline*ncol-len(ims))).reshape(nline,ncol)
    wtab= np.zeros_like(split_idx,dtype=int)
    htab= np.zeros_like(split_idx,dtype=int)
    
    for i in range(nline):
        w_tmp = 0
        for j in range(ncol):
            if np.isnan(split_idx[i,j]): continue
            elif ims[int(split_idx[i,j])] is None: continue
            else: wtab[i,j] = int(ims[int(split_idx[i,j])].width)
    w = wtab.max(axis=0).sum() + int(offset[0])*(wtab.shape[1]-1)
            
    for j in range(ncol):
        h_tmp = 0
        for i in range(nline):
            if np.isnan(split_idx[i,j]): continue
            elif ims[int(split_idx[i,j])] is None: continue
            else: htab[i,j] = int(ims[int(split_idx[i,j])].height)
    h = htab.max(axis=1).sum() + int(offset[1])*(htab.shape[0]-1)
    
    if title is not None:
        font = ImageFont.truetype(join(os.getenv('WINDIR'),'Fonts','Arial.ttf'), fontsize)
        tw,th = font.getsize(title)
        h += th
        dh = th
    else:
        dh = 0
    
    #Construct image
    dst = Image.new('RGB', (w, h), color=(255,255,255))
    
    for i in range(nline):
        dw = 0; dh_tmp = 0
        for j in range(ncol):
            if (wtab[i,j]>0) and (htab[i,j]>0):
                dst.paste(ims[int(split_idx[i,j])], (dw, dh))
            dw += wtab.max(axis=0)[j] + int(offset[0])
        dh += htab.max(axis=1)[i] + int(offset[1])
    
    #Add title
    if title is not None:
        draw = ImageDraw.Draw(dst)
        draw.text((int((w-tw)/2), 0),title,(0,0,0),font=font)
    
    return dst

if __name__ == '__main__':
    im1 = Image.open('image1.png')
    im2 = Image.open('image2.png')
    dst = concatPlot([im1,im2],nline=1,ncol=2,offset=[0,0],title='Title')
    dst.save('concat.png')
    
