#!/usr/bin/env python

from astropy.io import fits
from astropy    import wcs
import argparse
import numpy
import os

def getCoaddDir(rootDir, filter):

    if rootDir[-1] is '/':
        searchDir = rootDir + 'deepCoadd/' + filter.upper()
    else:
        searchDir = rootDir + '/deepCoadd/' + filter.upper()

    return searchDir


def listAllImages(rootDir, filter):

    import glob

    searchPattern = getCoaddDir(rootDir, filter) + '/*/*.fits'
    return map(lambda x: x, glob.glob(searchPattern))


def imgCornersRaDec(fileName, hdu=0, output=False):

    if not os.path.isfile(fileName):
        raise Exception('Can not find the image file: %s' % fileName)
    else:
        # Load the FITS Hdulist
        hduList = fits.open(fileName)
        # Get the header of the image HDU
        header  = hduList[hdu].header

    # Read the WCS information from the header
    w = wcs.WCS(header)
    # Get the dimension of the image
    xSize = header['NAXIS1']
    ySize = header['NAXIS2']

    # Define the Image coordinates of 4 corners
    pixLL = [0, 0]          # Lower left
    pixUL = [0, ySize]      # Upper left
    pixLR = [xSize, 0]      # Lower right
    pixUR = [xSize, ySize]  # Upper right

    # Numpy array
    pixCorners = numpy.array([pixLL, pixUL, pixLR, pixUR], numpy.float_)

    # Get the (RA, Dec) of four corners
    raDec = w.wcs_pix2world(pixCorners, 1)

    # Output results
    if output:
        print "%s , %12.7f , %12.7f , %12.7f , %12.7f , %12.7f , %12.7f , \
                %12.7f , %12.7f" % (fileName, raDec[0][0], raDec[0][1],
                                    raDec[1][0], raDec[1][1],
                                    raDec[2][0], raDec[2][1],
                                    raDec[3][0], raDec[3][1])

    return raDec

def coaddImgCornersRaDec(root, filt, prefix=None):

    # Get the image list
    imgList = listAllImages(root, filt)

    # Define the output FITS name
    if prefix is None:
        prefix = os.path.split(root)[-1].lower()
    outFits = prefix + '_' + filt + '_corners.fits'

    # Define all the columns
    files, rerun, filter, tract, patch = [], [], [], [], []
    raLL, decLL, raUL, decUL = [], [], [], []
    raLR, decLR, raUR, decUR = [], [], [], []

    for img in imgList:
        # Get rid of the '\n' at the end of the line
        imgFile = img.strip()
        files.append(imgFile)

        segFile = imgFile.split('/')
        if len(segFile) >= 5:
            rerun.append(segFile[-5])
            filter.append(segFile[-3])
            tract.append(int(segFile[-2]))
            patch.append(segFile[-1].split('.')[0])
        else:
            raise Exception("Image location is not correct!")

        # Get the corner coordinates
        imgRaDec = imgCornersRaDec(imgFile, hdu=1)
        # Lower left
        raLL.append(imgRaDec[0][0])
        decLL.append(imgRaDec[0][1])
        # Upper left
        raUL.append(imgRaDec[1][0])
        decUL.append(imgRaDec[1][1])
        # Lower right
        raLR.append(imgRaDec[2][0])
        decLR.append(imgRaDec[2][1])
        # Upper right
        raUR.append(imgRaDec[3][0])
        decUR.append(imgRaDec[3][1])

    columns = fits.ColDefs([
        fits.Column(name='file',  format='90A', array=numpy.array(files)),
        fits.Column(name='rerun', format='20A', array=numpy.array(rerun)),
        fits.Column(name='filter',format='5A', array=numpy.array(filter)),
        fits.Column(name='tract', format='I', array=numpy.array(tract)),
        fits.Column(name='patch', format='3A', array=numpy.array(patch)),
        fits.Column(name='raLL',  format='E', array=numpy.array(raLL),
                    unit='degree'),
        fits.Column(name='decLL', format='E', array=numpy.array(decLL),
                    unit='degree'),
        fits.Column(name='raUL',  format='E', array=numpy.array(raUL),
                    unit='degree'),
        fits.Column(name='decUL', format='E', array=numpy.array(decUL),
                    unit='degree'),
        fits.Column(name='raLR',  format='E', array=numpy.array(raLR),
                    unit='degree'),
        fits.Column(name='decLR', format='E', array=numpy.array(decLR),
                    unit='degree'),
        fits.Column(name='raUR',  format='E', array=numpy.array(raUR),
                    unit='degree'),
        fits.Column(name='decUR', format='E', array=numpy.array(decUR),
                    unit='degree')])

    tabHdu = fits.TableHDU.from_columns(columns)
    priHdu = fits.PrimaryHDU([0])
    hduList = fits.HDUList([priHdu, tabHdu])
    hduList.writeto(outFits, overwrite=True)

    return columns

def coaddAllBandsCorners(root, prefix=None):

    hscFilters = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']

    multiTable = []
    for filt in hscFilters:
        filtDir = getCoaddDir(root, filt)
        print "### Deal with %s band data at %s" % (filt, filtDir)
        if os.path.exists(filtDir):
            cornerTable = coaddImgCornersRaDec(root, filt, prefix=prefix)
            multiTable.append(cornerTable)
        else:
            print "### Folder for %s band data is not available" % filt

    return multiTable

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    #parser.add_argument("filt", help="HSC filter")
    parser.add_argument('-p', '--prefix', dest='prefix',
                        help='Prefix of the output file',
                        default=None)
    args = parser.parse_args()

    #coaddImgCornersRaDec(args.root, args.filt, prefix=args.prefix)
    coaddAllBandsCorners(args.root, prefix=args.prefix)
