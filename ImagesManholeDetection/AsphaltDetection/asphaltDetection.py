#!/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import array
import constant
import gc
import gdal
from osgeo import ogr, osr
from outils import getExtent

#from memory_profiler import profile

import constant

class asphaltDetection(object):

#    @profile
    def __init__ (self, nom_image):
        super(asphaltDetection,self).__init__()
        self.nom_image = nom_image    
        self.image = cv2.imread(constant.REP+nom_image)
        if (self.image.any() == None):
            print ("unable to load image")
            print (constant.REP+nom_image)
            sys.exit("Program aborted")

        self.height, self.width = self.image.shape[:2]
        self.n_pixels = self.height*self.width
        self.mask = np.zeros((self.height, self.width),np.uint8)
        self.CCLabels = np.zeros((self.height, self.width))
        self.n_labels = 0
        self.roots = 0

#    @profile
    def setRoots(self):
        self.roots = array.array('i',(0,)*self.n_pixels) #[[] for x in range(self.n_pixels)]

#    @profile
    def setBackground(self):
        for y in range(self.height):
            for x in  range(self.width):
                if (self.image[y,x].any==0): self.image[y,x]=constant.BACKGROUND

#    @profile
    def cutImage(self):
        list_image = []
        if (self.height > constant.MAXSIZE) and (self.width > constant.MAXSIZE):
            ny = int(self.height/constant.CUTSIZE)
            nx = int(self.width/constant.CUTSIZE)
            im_size_y = int(self.height)/ny
            im_size_x = int(self.width) /nx

            for j in range(ny):
                for i in range(nx):

                    out = self.image[im_size_y*j:(j+1)*im_size_y, im_size_x*i:(i+1)*im_size_x]
                    nom_image = "SUB_{0}{1}.tiff".format(j,i)
                    
                    if (out.mean() > 5):
                        self.createGeoTiffImage(constant.REP+self.nom_image, constant.REP+nom_image, out, im_size_x, im_size_y, im_size_x*i, im_size_y*j)
                        list_image.append(nom_image)
                        print ("image "+nom_image+" wrote")
                    else:
                        print ("skip image "+nom_image+": too dark to be treated")
        else :
            list_image = [self.nom_image]
                        
        return list_image
                        
#    @profile
    def inverseMask(self):
        self.mask = 255 - self.mask


#    @profile
    def maskImage(self):
        self.image[:,:,0] = self.image[:,:,0]*(self.mask/255.)
        self.image[:,:,1] = self.image[:,:,1]*(self.mask/255.)
        self.image[:,:,2] = self.image[:,:,2]*(self.mask/255.)


#    @profile
    def buildAsphaltMask(self):
        for y in range(self.height):
            for x in  range(self.width):
                px = self.image[y,x]
                #                m = px.mean() # semble + long...
                m = ( int(px[0]) + int(px[1]) + int(px[2]) )/3.
                thresh = 0.1*m
                if (m>50 and m<200) and (abs(px[0]-m) < thresh and abs(px[1]-m) < thresh and abs(px[2]-m) < thresh):
                    self.mask[y,x] = 255
#                if (m>50 and m<230) and (px.std() < 0.2*m):
# semble plus long...

    
#    @profile
    def find(self,pos):
    #    print ("in find", pos, self.roots[pos])
        while (self.roots[pos] != pos):
            pos = self.roots[pos]
        return pos

#    @profile
    def union(self,root0,root1):
        if(root0 == root1):
            return root0
        if(root0 == constant.NOROOT):
            return root1
        if(root1 == constant.NOROOT):
            return root0
        if(root0 < root1):
            self.roots[root1] = root0
            return root0;
        else:
            self.roots[root0] = root1
            return root1

#    @profile
    def add(self,pos,root):
        # format for the documentation
        """this function set the root of the node at position pos
            
        set to constant.BACKGROUND if root==constant.BACKGROUND  """
        pass
    #    print (max(self.roots))
        if (root==constant.NOROOT):
            self.roots[pos] = pos
        elif (root==constant.BACKGROUND):
            self.roots[pos] = constant.BACKGROUND
        else:
            self.roots[pos] = root

        return self.roots

#    @profile
    def unionFind(self):
        """This function call the Union and Find algorithm
            
            This function call the Union and Find algorithm
            to label the connected componants. Call removeSmallCC(min_size)
            to remove the CC containing less than min_size pixels """
        pos = 0
        for y in range(self.height):
            for x in range(self.width):
    
                root = constant.NOROOT
                px = self.mask[y,x]
    
                if px > 0:
                    if (x > 0 and (self.mask[y,x-1]==px)):
                        root = self.union(self.find(pos-1),root)
    
                    if( (x > 0) and (y > 0) and (self.mask[y-1,x-1] == px)):
                        root = self.union(self.find(pos-1-self.width), root)

                    if( (y > 0) and (self.mask[y-1,x] == px) ):
                        root = self.union(self.find(pos-self.width), root)

                    if( (x < self.width - 1) and (y > 0) and (self.mask[y-1,x+1] == px) ):
                        root = self.union(self.find(pos+1-self.width), root)

                    self.roots = self.add(pos, root)
    
                else:
                    self.add(pos,constant.BACKGROUND)
            
                pos = pos+1

    

#    @profile
    def buildLabelArray(self):

        for pos in range(self.n_pixels):
            if(self.roots[pos] != constant.BACKGROUND):
                self.roots[pos] = self.find(pos)

        label = 1;
        for pos in range(self.n_pixels):
            if(self.roots[pos] == pos):
                self.roots[pos] = label
                label = label +1
            elif(self.roots[pos] != constant.BACKGROUND):
                self.roots[pos] = self.roots[self.roots[pos]]



        pos =-1
        for y in range(self.height):
            for x in range(self.width):
                pos = pos+1
                if (self.roots[pos] == constant.BACKGROUND):
                    self.CCLabels[y,x] = 0
                else:
                    self.CCLabels[y,x] = self.roots[pos]

        self.n_labels = label-1

#    @profile
    def getNPixelsLabel(self):
        n_pixels_label = np.zeros(self.n_labels+1)
        for y in range(self.height):
            for x in range(self.width):
                n_pixels_label[int(self.CCLabels[y,x])] = n_pixels_label[int(self.CCLabels[y,x])]+1


        return n_pixels_label

#    @profile
    def removeSmallCC(self,thresh):
        n_pixels_label = self.getNPixelsLabel()
        for y in range(self.height):
            for x in range(self.width):
                if (self.mask[y,x] > 0):
                    if (n_pixels_label[int(self.CCLabels[y,x])] < thresh):
                        self.mask[y,x] = 0
                    else:
                        self.mask[y,x] = 255


    def createGeoTiffImage(self, src, dst, image, im_size_x = None, im_size_y = None, x_corner = None, y_corner = None):
        """
            Créé une image GeoTiff avec la même projection qu'une image de référence.

            @type src : String
            @param src : Nom de l'image géo-référencée de référence.

            @type dst : String
            @param dst : Nom de l'image géo-référencée en sortie.

            @type image : numpy.ndarray
            @param image : Image de sortie.

            @type im_size_x : Entier
            @param im_size_x : Largeur de l'image.

            @type im_size_y : Entier
            @param im_size_y : Hauteur de l'image.

            @type x_corner : Entier
            @param x_corner : Position en x du coin supérieur gauche.

            @type y_corner : Entier
            @param y_corner : Position en y du coin supérieur gauche.
        """
        print ("in geotiff:")
        print(src)
        imageTif = gdal.Open(src)

        info = imageTif.GetGeoTransform()

        pixelSizeX = info[1]
        pixelSizeY = -info[5]
                
        cols = imageTif.RasterXSize
        rows = imageTif.RasterYSize

        if  im_size_x is None :
            im_size_x = cols

        if im_size_y is None :
            im_size_y = rows

        ext = getExtent(info,cols,rows)

        wkt_projection = imageTif.GetProjectionRef()

        del imageTif

        origineX = ext[0][0]
        origineY = ext[0][1]

        deltaX = abs(ext[0][0] - ext[1][0])
        deltaY = abs(ext[0][1] - ext[1][1])

        facteurX = cols/deltaX
        facteurY = rows/deltaY

        if  x_corner is None and y_corner is None:
            xGeo = origineX
            yGeo = origineY
        else :
            xGeo = origineX + ((x_corner)/facteurX)
            yGeo = origineY - ((y_corner)/facteurY)

        b,g,r = cv2.split(image)

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(dst, im_size_x, im_size_y, 3, gdal.GDT_Byte)
                        
        dataset.SetGeoTransform((xGeo, pixelSizeX, 0, yGeo, 0, -pixelSizeY))
        dataset.SetProjection(wkt_projection)

        dataset.GetRasterBand(1).WriteArray(r)
        dataset.GetRasterBand(3).WriteArray(b)
        dataset.GetRasterBand(2).WriteArray(g)
                        
        dataset.FlushCache() 

