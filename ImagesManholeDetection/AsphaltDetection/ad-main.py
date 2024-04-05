#!/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import os, glob
import argparse
import time
#Mettre le fichier outil en commun???
from outils import Fusion, Log
from asphaltDetection import asphaltDetection as ad

import constant

#from memory_profiler import profile

# RQ: imread: returns a matrix of size [height,width][0:2] with Blue Red Green channels (not RGB)
#@profile
def main(argv=sys.argv):
    """
        Main program for asphalt detection
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("site", nargs='+', action=Fusion, help="Nom du site.")
    args = parser.parse_args()
    print ("---> read image ")
    print (constant.REP+args.site)
    img = ad(args.site)
    print ("        image read <---")

    print ("---> cut image ")
    list_images = img.cutImage()
    nb_images = len(list_images)
    if (nb_images>0):
        print ("     {0} subimages created".format(nb_images))


    for im in range(nb_images):
        tij = time.time()
        if (nb_images!=0):
            input_name = list_images[im]
            print ("Treatment of image "+input_name)
            del img
            img = ad(input_name)
            output_name = "filtered_"+input_name
        else:
            print ("Treatment of image {0}".format(args.site))
            output_name = "filtered_"+args.site
        
        img.setRoots()
        
        print ("---> Compute mask + open and dilatation")
#            img.setBackground()
        t1 = time.time()
        print ("     Mask...")
        img.buildAsphaltMask()
        kernel = np.ones((5,5),np.uint8)
        print ("     Open morpho...")
        img.mask = cv2.morphologyEx(img.mask, cv2.MORPH_OPEN, kernel)
        t2 = time.time()
        print ("        done in {0}min<---".format((t2-t1)/60.))

        print ("---> Compute connected componants for roads")
        t1 = time.time()
        print ("     Union Find ...")
        img.unionFind()
        print ("     Build Label Array ...")
        img.buildLabelArray()
        print ("     {0} connected componants".format(img.n_labels))
        thresh = 50000
        print ("     Remove CC < {0} pixels".format(thresh))
        img.removeSmallCC(thresh)
        img.mask = cv2.dilate(img.mask,kernel,iterations = 2)
        t2 = time.time()
        print ("        done in {0}min<---".format((t2-t1)/60.))

        print ("---> Remove very small isolated areas")
        t1 = time.time()
        img.inverseMask()
        print ("     Union Find ...")
        img.unionFind()
        print ("     Build Label Array ...")
        img.buildLabelArray()
        print("     {0} connected componants".format(img.n_labels))
        thresh = 5000
        print ("     Remove CC < {0} pixels".format(thresh))
        try:
            img.removeSmallCC(thresh)
            img.inverseMask()
        except Exception as e:
            print (e)
            raise "STOP"

        t2 = time.time()
        print ("        done in {0}min<---".format((t2-t1)/60.))


    #    cv2.imshow('image',img.mask)
    #    cv2.waitKey(0) & 0xFF
    #    cv2.destroyAllWindows()
        print ("---> Save output file")
        t1 = time.time()
        print (output_name)
        img.maskImage()
        img.createGeoTiffImage(constant.REP+img.nom_image, constant.REP+output_name, img.image)
        # cv2.imwrite(output_name,img.image)
        t2 = time.time()
        print ("        done in {0}min<---".format((t2-t1)/60.))

        print ("******* everything done for"+input_name+" in {0}min ********".format((t2-tij)/60.))


                       
if __name__ == "__main__":
    sys.exit(main())


#labels = roots.reshape(image_width,image_height)



#cv2.imshow('image',roots)
#cv2.waitKey(0) & 0xFF
#cv2.destroyAllWindows()
#cv2.imwrite("../../TESTS/DATA/Test_CC.tiff",labels)


