from osgeo import gdal,osr,ogr
import numpy as np
import sys
import os

#----------------------------------------------------------------------------------------------
# Resample Geotiff Raster
def ResampleGeotiff(infname, outfname,minx, miny, maxx, maxy,dx,dy):
    box_str = "%f %f %f %f" % (minx, miny, maxx, maxy)
    pixsp_str = "%f %f" %(dx, dy)
    #UInt16
    command = 'gdalwarp -overwrite -te %s -tr %s -r max -ot byte -of Gtiff %s %s'% (box_str,pixsp_str, infname, outfname)     
    return os.system(f'{command:s}')


name_shp = "/Users/cdelenne/Documents/_DONNEES/MontpellierEtEnviron/Gigean/Shp_Gigean/ManholesOrthoAJ_Carole.shp"
name_image = "/Users/cdelenne/Documents/_DONNEES/MontpellierEtEnviron/Gigean/2015AvionJaune_Gigean/Georef/140610_gigean_4cm_l93_vis.tif"
outpath = "/Users/cdelenne/Documents/_DONNEES/MontpellierEtEnviron/Gigean/ThumbManholesAJCarole/"
EPSG = 0

shpDriver = ogr.GetDriverByName("ESRI Shapefile")
PointSHP = shpDriver.Open(name_shp, 1)
if PointSHP == None:
    raise Exception("Unable to open SHP ....................................", name_shp)

LayerPoints = PointSHP.GetLayer()
PointsCount = LayerPoints.GetFeatureCount()
print(PointsCount)

SpatialReference = osr.SpatialReference()
if (EPSG != 0) :
    SpatialReference.ImportFromEPSG(EPSG)
else:
    spatialRef = LayerPoints.GetSpatialRef() 
    SpatialReference.ImportFromWkt(spatialRef.ExportToWkt()) 	



# Read image
try:
    Image = gdal.Open(name_image)
except RuntimeError:
    print("Unable to open ", name_image)
    sys.exit(1)
    
        
try:
    ImBand=Image.GetRasterBand(1)
except RuntimeError:
    print("Unable to read", name_image)
    sys.exit(1)

gt = Image.GetGeoTransform()
rasterDriver = gdal.GetDriverByName('GTiff')

t=1
for P in LayerPoints :
    point = P.GetGeometryRef()
    xGeo = point.GetPoint()[0]
    yGeo = point.GetPoint()[1]
    
    outfname = outpath + "Manhole" + str(t) + ".tif"
    ResampleGeotiff(name_image, outfname, xGeo-1, yGeo-1, xGeo+1, yGeo+1, abs(gt[1]),abs(gt[5]))
    t=t+1
   
    