#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from logging.handlers import RotatingFileHandler

class Fusion(argparse.Action):
	"""
		Classe permettant d'avoir plusieurs mots comme un seul argument.
	"""
	
	def __init__(self, option_strings, dest, nargs=None, **kwargs):
		super(Fusion, self).__init__(option_strings, dest, nargs, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, ' '.join(values))

class Log(object):
	"""
		Classe de gestion des logs.
	"""

	def __init__(self, dossier, nomFichier, niveau=logging.DEBUG):
		super(Log, self).__init__()

		self.__logger__ = logging.getLogger(nomFichier)
		self.__logger__.setLevel(niveau)

		format = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
		fichierLog = RotatingFileHandler("{0}{1}.log".format(dossier, nomFichier), 'a', 1000000, 1)

		fichierLog.setLevel(niveau)
		fichierLog.setFormatter(format)
		self.__logger__.addHandler(fichierLog)

		console = logging.StreamHandler()
		console.setLevel(niveau)
		self.__logger__.addHandler(console)

	def info(self, message):
		self.__logger__.info(message)

	def debug(self, message):
		self.__logger__.debug(message)

	def warning(self, message):
		self.__logger__.warning(message)

	def error(self, message):
		self.__logger__.error(message)

	def critical(self, message):
		self.__logger__.critical(message)

	def close(self):
		"""
			Fermeture des logs.
		"""
		for handler in  self.__logger__.handlers[:] :
			handler.close()
			self.__logger__.removeHandler(handler)


def getExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()

    return [ext[0], ext[2]]