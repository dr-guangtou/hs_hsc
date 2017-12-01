#!/usr/bin/env python
# encoding: utf-8
"""Generate HSC cutout image for galaxy."""

from __future__ import (division, print_function)

import os
import copy
import fcntl
import argparse
import warnings

# HSC Pipeline
import lsst.daf.persistence as dafPersist
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable

import numpy as np

# Astropy
from astropy import wcs as apWcs
from astropy.io import fits

# Personal
import coaddColourImage as cdColor
import hscUtils as hUtil

# Matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
plt.ioff()
import matplotlib.patches as mpatches

# Default colormap
viridis = plt.get_cmap('viridis')
viridis.set_bad('k', 1.)
viridis_r = plt.get_cmap('viridis_r')

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100
PIXEL_SCALE = 0.168  # arcsec/ pixel

# About the Colormaps
SEG_CMAP = hUtil.random_cmap(ncolors=512,
                             background_color=u'white')
SEG_CMAP.set_bad(color='white')
SEG_CMAP.set_under(color='white')


def saveImageArr(arr, header, name, overwrite=True):
    """
    Just save an array to a fits file.

    Parameters:
    """
    hduImg = fits.PrimaryHDU(arr, header=header)
    hduList = fits.HDUList([hduImg])
    hduList.writeto(name, overwrite=overwrite)
    hduList.close()


def flatSrcArr(srcArr):
    """Flat a String Array."""
    for (ii, cat) in enumerate(srcArr):
        if ii == 0:
            srcUse = cat
        else:
            for (jj, item) in enumerate(cat):
                srcUse.append(item)
    return srcUse


def previewCoaddImage(img,
                      msk,
                      var,
                      det,
                      sizeX=10,
                      sizeY=10,
                      prefix='hsc_cutout',
                      outPNG=None,
                      oriX=None,
                      oriY=None,
                      boxW=None,
                      boxH=None):
    """
    Generate a preview figure.

    Parameters:
    """
    fig, axes = plt.subplots(sharex=True, sharey=True,
                             figsize=(sizeX, sizeY))
    # Image
    imin, imax = hUtil.zscale(np.arcsinh(img), contrast=0.1,
                              samples=500)
    imax += 0.001

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(
        np.arcsinh(img),
        interpolation="none",
        vmin=imin,
        vmax=imax,
        cmap=viridis,
        origin='lower')
    ax1.minorticks_on()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.text(
        0.5,
        0.9,
        'Coadd Image',
        fontsize=25,
        fontweight='bold',
        ha='center',
        va='center',
        color='w',
        transform=ax1.transAxes)

    # Put scale bar on the image
    hsc_pixels_20asec = (20.0 / PIXEL_SCALE)
    (img_size_x, img_size_y) = img.shape
    scale_bar_x_0 = int(img_size_x * 0.95 - hsc_pixels_20asec)
    scale_bar_x_1 = int(img_size_x * 0.95)
    scale_bar_y = int(img_size_y * 0.10)
    scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
    scale_bar_text_y = (scale_bar_y * 0.50)
    scale_bar_text = '20 arcsec'
    scale_bar_text_size = 13

    ax1.plot([scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
             linewidth=3, c='w', alpha=1.0)
    ax1.text(scale_bar_text_x, scale_bar_text_y, scale_bar_text,
             fontsize=scale_bar_text_size,
             horizontalalignment='center',
             color='w')

    ax1.margins(0.00, 0.00, tight=True)

    # Variance
    smin, smax = hUtil.zscale(np.arcsinh(var),
                              contrast=0.25, samples=500)
    smax += 0.01

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(
        np.arcsinh(var),
        interpolation="none",
        vmin=smin,
        vmax=smax,
        cmap=viridis,
        origin='lower')

    ax2.minorticks_on()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.text(
        0.5,
        0.9,
        'Variance',
        fontsize=25,
        fontweight='bold',
        ha='center',
        va='center',
        color='w',
        transform=ax2.transAxes)

    # Mask
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(det, cmap=plt.cm.gray, alpha=0.8, origin='lower')
    ax3.imshow(msk, cmap=SEG_CMAP, alpha=0.8, origin='lower')

    ax3.minorticks_on()
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.text(
        0.5,
        0.9,
        'Mask Plane',
        fontsize=25,
        fontweight='bold',
        ha='center',
        va='center',
        color='k',
        transform=ax3.transAxes)

    # If necessary, outline the BBox of each overlapped Patch
    if ((oriX is not None) and (oriY is not None) and (boxW is not None) and
            (boxH is not None)):
        numBox = len(oriX)
        for ii in range(numBox):
            box = mpatches.Rectangle(
                (oriX[ii], oriY[ii]),
                boxW[ii],
                boxH[ii],
                fc="none",
                ec='r',
                linewidth=3.5,
                alpha=0.6,
                linestyle='dashed')
            ax3.add_patch(box)

    # Masked Image
    imgMsk = copy.deepcopy(img)
    imgMsk[(msk > 0) | (det > 0)] = np.nan

    mmin, mmax = hUtil.zscale(np.arcsinh(img), contrast=0.75, samples=500)
    mmax += 0.01
    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(
        np.arcsinh(imgMsk),
        interpolation="none",
        vmin=mmin,
        vmax=mmax,
        cmap=viridis,
        origin='lower')
    ax4.minorticks_on()
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.text(
        0.5,
        0.9,
        'Masked Image',
        fontsize=25,
        fontweight='bold',
        ha='center',
        va='center',
        color='w',
        transform=ax4.transAxes)

    # Adjust the figure
    plt.subplots_adjust(
        hspace=0.0, wspace=0.0, bottom=0.0, top=1.0,
        right=1.0, left=0.0)

    # Save a PNG file
    if outPNG is None:
        outPNG = prefix + '_pre.png'
    plt.savefig(outPNG, dpi=70)
    plt.close(fig)


def getCoaddPsfImage(calExp, coord, label="None"):
    """Get the coadd PSF image."""
    # Get the WCS information
    wcs = calExp.getWcs()
    # The X,Y coordinate of the image center
    coordXY = wcs.skyToPixel(coord)
    # Get the PSF object for the exposure
    psf = calExp.getPsf()
    try:
        psfImg = psf.computeImage(coordXY)
        return psfImg
    except Exception:
        if prefix is "None":
            warnings.warn("### Can not compute PSF Image !!!")
            print("\n !!! Can not compute PSF Image !!!")
        else:
            warnings.warn("### Can not compute PSF Image : %s !!!" % label)
            print("\n !!! Can not compute PSF Image : % s!!!" % label)
        return None


def getCoaddMskPlane(calExp, bitmask):
    """Get the mask plane of the coadd.

    These are the hscPipe 5.4 bitmasks:
        Plane 0 -> BAD
        Plane 1 -> SAT
        Plane 2 -> INTRP
        Plane 3 -> CR
        Plane 4 -> EDGE
        Plane 5 -> DETECTED
        Plane 6 -> DETECTED_NEGATIVE
        Plane 7 -> SUSPECT
        Plane 8 -> NO_DATA
        Plane 9 -> BRIGHT_OBJECT
        Plane 10 -> CLIPPED
        Plane 11 -> CROSSTALK
        Plane 12 -> NOT_DEBLENDED
        Plane 13 -> UNMASKEDNAN
    """
    # Get the mask image
    mskImg = calExp.getMaskedImage().getMask()
    newMsk = copy.deepcopy(mskImg)
    try:
        # Extract specific plane from it
        newMsk &= newMsk.getPlaneBitMask(bitmask)
    except Exception:
        mskImg.printMaskPlanes()
        newMsk = None

    return newMsk


def getCoaddBadMsk(calExp, no_bright_object=False):
    """Get the BAD mask plane."""
    mskImg = calExp.getMaskedImage().getMask()

    badMsk = copy.deepcopy(mskImg)
    # Clear the "DETECTED" plane
    badMsk.removeAndClearMaskPlane('DETECTED', True)

    try:
        # Clear the "DETECTED_NEGATIVE" plane
        badMsk.removeAndClearMaskPlane('DETECTED_NEGATIVE', True)
    except Exception:
        pass

    # try:
    #     # Clear the "CLIPPED" plane
    #     badMsk.removeAndClearMaskPlane('CLIPPED', True)
    # except Exception:
    #     pass

    try:
        # Clear the "CROSSTALK" plane
        badMsk.removeAndClearMaskPlane('CROSSTALK', True)
    except Exception:
        pass

    try:
        # Clear the "NOT_DEBLENDED" plane
        badMsk.removeAndClearMaskPlane('NOT_DEBLENDED', True)
    except Exception:
        pass

    if no_bright_object:
        try:
            # Clear the "BRIGHT_OBJECT" plane
            badMsk.removeAndClearMaskPlane('BRIGHT_OBJECT', True)
        except Exception:
            pass

    return badMsk


def getCircleRaDec(ra, dec, size):
    """
    Get the (RA, DEC) that describe a circle.

    Region around the central input coordinate
    """
    # Convert the size from pixel unit to degress
    sizeDegree = (size * 0.168) / 3600.0
    # representative set of polar angles
    angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    phi = np.array(angles * np.pi / 180.0)

    raList = ra + sizeDegree * np.cos(phi)
    decList = dec + sizeDegree * np.sin(phi)

    return raList, decList


def coaddImageCutout(root,
                     ra,
                     dec,
                     size,
                     saveMsk=True,
                     saveSrc=True,
                     filt='HSC-I',
                     prefix='hsc_coadd_cutout',
                     circleMatch=True,
                     verbose=True,
                     extraField1=None,
                     extraValue1=None,
                     butler=None,
                     no_bright_object=False):
    """Cutout coadd image around a RA, DEC."""
    # No longer support hscPipe < 4
    coaddData = "deepCoadd_calexp"

    # See if we are using hscPipe > 5
    try:
        dafPersist.eupsVersions.EupsVersions().versions['hscPipe']
        hscPipe5 = False
    except AttributeError:
        hscPipe5 = True

    # Get the SkyMap of the database
    if butler is None:
        try:
            butler = dafPersist.Butler(root)
            if verbose:
                print(SEP)
                print("## Load in the Butler ...")
        except Exception:
            raise Exception("### Can not load the Butler")
    skyMap = butler.get("deepCoadd_skyMap", immediate=True)

    # Get the expected cutout size
    dimExpect = (2 * size + 1)
    sizeExpect = dimExpect**2
    # Cutout size in unit of degree
    sizeDeg = size * 0.168 / 3600.0

    # Verbose
    if verbose:
        print(SEP)
        print(" Input Ra, Dec: %10.5f, %10.5f" % (ra, dec))
        print(" Cutout size is expected to be %d x %d" % (dimExpect,
                                                          dimExpect))

    # First, search for the central (Ra, Dec)
    # Define the Ra, Dec pair
    coord = afwCoord.IcrsCoord(ra * afwGeom.degrees, dec * afwGeom.degrees)

    # Search for overlapped tract, patch pairs
    matches = skyMap.findClosestTractPatchList([coord])
    # Number of matched tracts
    nTract = len(matches)
    # Number of matched (patches)
    nPatch = 0
    for tt in range(nTract):
        nPatch += len(matches[tt][1])
    if verbose:
        print("## Find %d possible matches !" % nPatch)

    matchCen = []
    for tract, patch in matches:

        # Get the (tract, patch) ID
        tractId = tract.getId()
        patchId = "%d,%d" % patch[0].getIndex()
        if verbose:
            print("## Choose (Tract, Patch) for center: %d, %s !" % (tractId,
                                                                     patchId))
        matchCen.append((tractId, patchId))
        # Get the coadd images
        # Try to load the coadd Exposure; the skymap covers larger area than
        # available data, which will cause Butler to fail sometime
        try:
            coadd = butler.get(
                coaddData,
                tract=tractId,
                patch=patchId,
                filter=filt,
                immediate=True)
        except Exception, errMsg:
            print(WAR)
            print(" The desired coordinate is not available !!! ")
            print(WAR)
            print(errMsg)
            """ TODO """
            coaddFound = False
            noData = True
            partialCut = True
            continue
        else:
            """
            It's still possible that the matched location actually has no
            useful data (e.g. have been interpolated, or in the NO_DATA
            part of the patch)
            For this situation, no psf image can be generated !!
            """
            coaddFound = True
            # Get the Coadded PSF image
            psfImg = getCoaddPsfImage(coadd, coord, label=prefix)
            if psfImg is None:
                noData = True
                partialCut = True
                continue
            else:
                noData = False
            # Get the WCS information
            wcs = coadd.getWcs()
            # Convert the central coordinate from Ra,Dec to pixel unit
            pixel = wcs.skyToPixel(coord)
            pixel = afwGeom.Point2I(pixel)
            # Define the bounding box for the central pixel
            bbox = afwGeom.Box2I(pixel, pixel)
            # Grow the bounding box to the desired size
            bbox.grow(int(size))
            # Compare to the coadd image, and clip
            bbox.clip(coadd.getBBox(afwImage.PARENT))
            if bbox.isEmpty():
                noData = True
                partialCut = True
                continue
            else:
                if bbox.getArea() < sizeExpect:
                    partialCut = True
                    if verbose:
                        print("## Cut out image dimension " +
                              "is : %d x %d " % (bbox.getWidth(),
                                                 bbox.getHeight()))
                else:
                    partialCut = False
            # Make a new ExposureF object for the cutout region
            subImage = afwImage.ExposureF(coadd, bbox, afwImage.PARENT)
            # Get the WCS
            subWcs = subImage.getWcs()
            # Get the central pixel coordinates on new subImage WCS
            newX, newY = subWcs.skyToPixel(coord)
            # Get the new origin pixel
            newOriX, newOriY = subImage.getImage().getXY0()
            newX = newX - newOriX
            newY = newY - newOriY
            # Get the header of the new subimage
            subHead = subImage.getMetadata()
            subHead.set('RA_CUT', ra)
            subHead.set('DEC_CUT', dec)
            subHead.set('NEW_X', newX)
            subHead.set('NEW_Y', newY)
            if partialCut:
                subHead.set('PARTIAL', 1)
            else:
                subHead.set('PARTIAL', 0)
            if (extraField1 is not None) and (extraValue1 is not None):
                subHead.set(extraField1, extraValue1)

            # To see if data are available for all the cut-out region
            if partialCut:
                warnings.warn("## Only part of the region is available" +
                              " : %d, %s" % (tractId, patchId))
                outPre = (prefix + '_' + str(tractId) + '_' + patchId + '_' +
                          filt + '_cent')
            else:
                outPre = (prefix + '_' + str(tractId) + '_' + patchId + '_' +
                          filt + '_full')
            # Define the output file name
            outImg = outPre + '.fits'
            outPsf = outPre + '_psf.fits'
            # Save the cutout image to a new FITS file
            subImage.writeFits(outImg)
            # Save the PSF image
            psfImg.writeFits(outPsf)
            if saveMsk is True:
                # Get the "Bad" mask plane
                mskBad = getCoaddBadMsk(subImage,
                                        no_bright_object=no_bright_object)
                mskBad.writeFits(outPre + '_bad.fits')

            if saveSrc is True:

                # Get the forced photometry source catalog
                """ Sometimes the forced photometry catalog is missing """
                try:
                    # TODO: Maybe the measurement catalog is better
                    srcCat = butler.get(
                        'deepCoadd_meas',
                        tract=tractId,
                        patch=patchId,
                        filter=filt,
                        immediate=True,
                        flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
                    # Get the pixel coordinates for all objects
                    srcRa = np.array(
                        map(lambda x: x.get('coord').getRa().asDegrees(),
                            srcCat))
                    srcDec = np.array(
                        map(lambda x: x.get('coord').getDec().asDegrees(),
                            srcCat))
                    # Simple Box match
                    indMatch = ((srcRa > (ra - sizeDeg)) & (srcRa <
                                                            (ra + sizeDeg)) &
                                (srcDec > (dec - sizeDeg)) & (srcDec <
                                                              (dec + sizeDeg)))
                    # Extract the matched subset
                    srcMatch = srcCat.subset(indMatch)
                    # Save the src catalog to a FITS file
                    outSrc = outPre + '_src.fits'
                    srcMatch.writeFits(outSrc)
                except Exception:
                    print("### Tract: %d  Patch: %s" % (tractId, patchId))
                    warnings.warn("### Can not find the *force catalog !")
                    noSrcFile = prefix + '_nosrc_' + filt + '.lis'
                    if not os.path.isfile(noSrcFile):
                        os.system('touch ' + noSrcFile)
                    with open(noSrcFile, "a") as noSrc:
                        try:
                            noSrc.write("%d  %s \n" % (tractId, patchId))
                            fcntl.flock(noSrc, fcntl.LOCK_UN)
                        except IOError:
                            pass

    # If only part of the desired cutout region is covered, and the circleMatch
    # Flag is set, find all the Patches that overlap with a circle region
    # around the input Ra, Dec
    if partialCut and circleMatch:

        if verbose:
            print("####### Search for other overlapped patches #######")

        # Return the list of RA, DEC that described a circle region around the
        # input (RA, DEC).  The radius is the input size in unit of arcsec
        raList, decList = getCircleRaDec(ra, dec, (size * 0.8))
        points = map(lambda x, y: afwGeom.Point2D(x, y), raList, decList)
        coords = map(lambda x: afwCoord.IcrsCoord(x), points)

        # Search for overlapped tract, patch pairs
        matches = skyMap.findTractPatchList(coords)
        # Number of matched tracts
        nTract = len(matches)
        # Number of matched (patches)
        nPatch = 0
        for tt in range(nTract):
            nPatch += len(matches[tt][1])
        if verbose:
            print("### Find %d possible overlap patches" % nPatch)

        for tract, patch in matches:

            # Example all possible matches
            tractId = tract.getId()
            for ii in range(len(patch)):
                patchId = "%d,%d" % patch[ii].getIndex()

                # Skip the image that has been used
                if (tractId, patchId) in matchCen:
                    if verbose:
                        print("### %d - %s is used" % (tractId, patchId))
                    continue

                # Get the coadd images
                # Try to load the coadd Exposure; the skymap covers
                # larger area than the available data, which will
                # cause Butler to fail sometime
                try:
                    coadd = butler.get(
                        coaddData,
                        tract=tractId,
                        patch=patchId,
                        filter=filt,
                        immediate=True)

                except Exception, errMsg:

                    print("#############################################")
                    print("              PARTIAL OVERLAP                ")
                    print(" The desired coordinate is not available !!! ")
                    print("#############################################")
                    print(errMsg)

                else:
                    # Get the WCS information
                    wcs = coadd.getWcs()

                    # Convert the central coordinate from Ra,Dec to pixel unit
                    pixel = wcs.skyToPixel(coord)
                    pixel = afwGeom.Point2I(pixel)

                    # Define the bounding box for the central pixel
                    bbox = afwGeom.Box2I(pixel, pixel)

                    # Grow the bounding box to the desired size
                    bbox.grow(int(size))

                    # Compare to the coadd image, and clip
                    bbox.clip(coadd.getBBox(afwImage.PARENT))

                    if bbox.isEmpty():
                        continue
                    elif bbox.getArea() < int(sizeExpect * 0.1):
                        # Ignore small overlapped image
                        if verbose:
                            print(WAR)
                            print("### %d-%s : small overlap" % (tractId,
                                                                 patchId))
                        continue
                    else:
                        if verbose:
                            print(WAR)
                            print("### Find one overlap: %d, %s" % (tractId,
                                                                    patchId))

                    # Make a new ExposureF object for the cutout region
                    subImage = afwImage.ExposureF(coadd, bbox, afwImage.PARENT)
                    # Get the WCS
                    subWcs = subImage.getWcs()
                    # Get the central pixel coordinates on new subImage WCS
                    newX, newY = subWcs.skyToPixel(coord)
                    # Get the new origin pixel
                    newOriX, newOriY = subImage.getImage().getXY0()
                    newX = newX - newOriX
                    newY = newY - newOriY

                    # Get the header of the new subimage
                    subHead = subImage.getMetadata()
                    subHead.set('RA_CUT', ra)
                    subHead.set('DEC_CUT', dec)
                    subHead.set('NEW_X', newX)
                    subHead.set('NEW_Y', newY)
                    if (extraField1 is not None) and (extraValue1 is not None):
                        subHead.set(extraField1, extraValue1)

                    # To see if data are available for all the cut-out region
                    outPre = (prefix + '_' + str(tractId) + '_' + patchId + '_'
                              + filt + '_part')
                    # Define the output file name
                    outImg = outPre + '.fits'
                    # Save the cutout image to a new FITS file
                    subImage.writeFits(outImg)

                    if saveMsk is True:
                        # Get different mask planes
                        mskDetec = getCoaddMskPlane(subImage, 'DETECTED')
                        mskIntrp = getCoaddMskPlane(subImage, 'INTRP')
                        mskSatur = getCoaddMskPlane(subImage, 'SAT')
                        mskDetec.writeFits(outPre + '_detec.fits')
                        mskIntrp.writeFits(outPre + '_intrp.fits')
                        mskSatur.writeFits(outPre + '_satur.fits')

                        # Get the "Bad" mask plane
                        mskBad = getCoaddBadMsk(subImage)
                        mskBad.writeFits(outPre + '_bad.fits')

    return coaddFound, noData, partialCut


def coaddImageCutFull(root,
                      ra,
                      dec,
                      size,
                      saveSrc=True,
                      savePsf=True,
                      filt='HSC-I',
                      prefix='hsc_coadd_cutout',
                      verbose=True,
                      extraField1=None,
                      extraValue1=None,
                      butler=None,
                      visual=True,
                      imgOnly=False,
                      no_bright_object=False):
    """Get the cutout around a location."""
    # No longer support hscPipe < 4
    coaddData = "deepCoadd_calexp"

    # See if we are using hscPipe > 5
    try:
        dafPersist.eupsVersions.EupsVersions().versions['hscPipe']
        hscPipe5 = False
    except AttributeError:
        hscPipe5 = True

    # Get the SkyMap of the database
    if butler is None:
        try:
            butler = dafPersist.Butler(root)
        except Exception:
            print('## Can not load the correct Butler!')

    skyMap = butler.get("deepCoadd_skyMap", immediate=True)

    # Check the filter
    if not cdColor.isHscFilter(filt, short=False):
        raise Exception("## Wrong Filter for HSC Data!")

    # (Ra, Dec) Pair for the center
    raDec = afwCoord.Coord(ra * afwGeom.degrees, dec * afwGeom.degrees)
    # [Ra, Dec] list
    raList, decList = cdColor.getCircleRaDec(ra, dec, size)
    points = map(lambda x, y: afwGeom.Point2D(x, y), raList, decList)
    raDecList = map(lambda x: afwCoord.IcrsCoord(x), points)

    # Expected size and center position
    dimExpect = int(2 * size + 1)
    cenExpect = (dimExpect / 2.0, dimExpect / 2.0)
    sizeExpect = int(dimExpect**2)
    # Get the half size of the image in degree
    sizeDegree = size * 0.168 / 3600.0

    # Verbose
    if verbose:
        print(SEP)
        print(" Input Ra, Dec: %10.5f, %10.5f" % (ra, dec))
        print(" Cutout size is expected to be %d x %d" % (dimExpect,
                                                          dimExpect))

    # Create empty arrays
    imgEmpty = np.empty((dimExpect, dimExpect), dtype="float")
    imgEmpty.fill(np.nan)
    if not imgOnly:
        mskEmpty = np.empty((dimExpect, dimExpect), dtype="uint8")
        varEmpty = np.empty((dimExpect, dimExpect), dtype="float")
        detEmpty = np.empty((dimExpect, dimExpect), dtype="float")
        mskEmpty.fill(np.nan)
        varEmpty.fill(np.nan)
        detEmpty.fill(np.nan)

    # Figure out the area we want, and read the data.
    # For coadds the WCS is the same in all bands,
    # but the code handles the general case
    # Start by finding the tract and patch
    matches = skyMap.findTractPatchList(raDecList)
    tractList, patchList = cdColor.getTractPatchList(matches)
    nPatch = len(patchList)
    if verbose:
        print("### Will deal with %d patches" % nPatch)

    # Prefix of the output file
    outPre = prefix + '_' + filt + '_full'

    newX = []
    newY = []
    boxX = []
    boxY = []
    boxSize = []
    #
    trList = []
    paList = []
    zpList = []
    #
    imgArr = []
    mskArr = []
    varArr = []
    detArr = []
    psfArr = []
    #
    srcArr = []
    refArr = []
    forceArr = []

    # Go through all these images
    for j in range(nPatch):
        # Tract, patch
        tract, patch = tractList[j], patchList[j]
        print("### Dealing with %d - %s" % (tract, patch))
        print(SEP)
        # Check if the coordinate is available in all three bands.
        try:
            # Get the coadded exposure
            coadd = butler.get(
                coaddData,
                tract=tract,
                patch=patch,
                filter=filt,
                immediate=True)
        except Exception, errMsg:
            print(WAR)
            print(" No data is available in %d - %s" % (tract, patch))
            print(WAR)
        else:
            # Get the WCS information
            wcs = coadd.getWcs()
            # Check if cdMatrix has been assigned
            cdExist = 'cdMatrix' in locals()
            if not cdExist:
                # Get the CD Matrix of the WCS
                cdMatrix = wcs.getCDMatrix()
                # Get the pixel size in arcsec
                pixScale = wcs.pixelScale().asDegrees() * 3600.0
                # Get the total exposure time
                visitIn = coadd.getInfo().getCoaddInputs().visits
                ccdIn = coadd.getInfo().getCoaddInputs().ccds

            # Convert the central coordinate from Ra,Dec to pixel unit
            pixel = wcs.skyToPixel(raDec)
            pixel = afwGeom.Point2I(pixel)
            # Define the bounding box for the central pixel
            bbox = afwGeom.Box2I(pixel, pixel)
            # Grow the bounding box to the desired size
            bbox.grow(int(size))
            xOri, yOri = bbox.getBegin()
            # Compare to the coadd image, and clip
            bbox.clip(coadd.getBBox(afwImage.PARENT))

            # Get the masked image
            try:
                subImage = afwImage.ExposureF(coadd, bbox, afwImage.PARENT)
            except Exception:
                print(WAR)
                print('### SOMETHING IS WRONG WITH THIS BOUNDING BOX !!')
                print("    %d -- %s -- %s " % (tract, patch, filt))
                print("    Bounding Box Size: %d" % (
                    bbox.getWidth() * bbox.getHeight()))
            else:
                # Extract the image array
                imgArr.append(subImage.getMaskedImage().getImage().getArray())

                if not imgOnly:
                    # Extract the detect mask array
                    mskDet = getCoaddMskPlane(subImage, 'DETECTED')
                    detArr.append(mskDet.getArray())

                    # Extract the variance array
                    imgVar = subImage.getMaskedImage().getVariance().getArray()
                    varArr.append(imgVar)

                    # Extract the bad mask array
                    mskBad = getCoaddBadMsk(subImage,
                                            no_bright_object=no_bright_object)
                    mskArr.append(mskBad.getArray())
                    # Get the source catalog
                    if saveSrc:
                        noFootprint = afwTable.SOURCE_IO_NO_FOOTPRINTS
                        print("### Search the source catalog....")
                        """
                        !!! Sometimes the forced photometry catalog
                            might not be available
                        """
                        try:
                            print("    !!!! TRY deepCoadd_meas")
                            srcCat = butler.get(
                                'deepCoadd_meas',
                                tract=tract,
                                patch=patch,
                                filter=filt,
                                immediate=True,
                                flags=noFootprint)
                            # Get the pixel coordinates for all objects
                            srcRa = np.array(
                                    map(lambda x:
                                        x.get('coord').getRa().asDegrees(),
                                        srcCat)
                                    )
                            srcDec = np.array(
                                    map(lambda x:
                                        x.get('coord').getDec().asDegrees(),
                                        srcCat)
                                    )
                            # Simple Box match
                            indMatch = ((srcRa > (ra - sizeDegree)) &
                                        (srcRa < (ra + sizeDegree)) &
                                        (srcDec > (dec - sizeDegree)) &
                                        (srcDec < (dec + sizeDegree)) &
                                        (srcCat.get('detect.is-patch-inner')))
                            # Extract the matched subset
                            srcArr.append(srcCat.subset(indMatch))
                            srcFound = True

                            # Try other catalogs
                            # 1. Reference
                            try:
                                print("    !!!! TRY deepCoadd_ref")
                                refCat = butler.get(
                                    'deepCoadd_ref',
                                    tract=tract,
                                    patch=patch,
                                    filter=filt,
                                    immediate=True,
                                    flags=noFootprint)
                                refArr.append(refCat.subset(indMatch))
                            except Exception:
                                warnings.warn('### No *ref catalog!')
                            # 2. Forced Photometry
                            try:
                                print("    !!!! TRY deepCoadd_forced_src")
                                forceCat = butler.get(
                                    'deepCoadd_forced_src',
                                    tract=tract,
                                    patch=patch,
                                    filter=filt,
                                    immediate=True,
                                    flags=noFootprint)
                                forceArr.append(forceCat.subset(indMatch))
                            except Exception:
                                warnings.warn('### No *force catalog!')
                        except Exception:
                            print("### Tract: %d  Patch: %s" % (tract, patch))
                            warnings.warn("### No photometry catalog!")
                            noSrcFile = prefix + '_nosrc_' + filt + '.lis'
                            if not os.path.isfile(noSrcFile):
                                dum = os.system('touch ' + noSrcFile)

                            with open(noSrcFile, "a") as noSrc:
                                try:
                                    noSrc.write("%d  %s \n" % (tract, patch))
                                    fcntl.flock(noSrc, fcntl.LOCK_UN)
                                except IOError:
                                    pass

                            srcFound = False
                # Save the width of the BBox
                boxX.append(bbox.getWidth())
                # Save the heigth of the BBox
                boxY.append(bbox.getHeight())
                # Save the size of the BBox in unit of pixels
                boxSize.append(bbox.getWidth() * bbox.getHeight())
                # New X, Y origin coordinates
                newX.append(bbox.getBeginX() - xOri)
                newY.append(bbox.getBeginY() - yOri)
                # Tract, Patch
                trList.append(tract)
                paList.append(patch)
                # Photometric zeropoint
                zpList.append(
                    2.5 * np.log10(coadd.getCalib().getFluxMag0()[0]))

                # If necessary, save the psf images
                if savePsf and (not imgOnly):
                    psfImg = getCoaddPsfImage(coadd, raDec,
                                              label=(str(tract).strip() +
                                                     str(patch))
                                              )
                    psfArr.append(psfImg)
                # Get the new (X,Y) coordinate of the galaxy center
                newCenExist = 'newCenX' in locals() and 'newCenY' in locals()
                if not newCenExist:
                    subWcs = subImage.getWcs()
                    newCenX, newCenY = subWcs.skyToPixel(raDec)
                    newCenX = newCenX - xOri
                    newCenY = newCenY - yOri

    # Number of returned images
    nReturn = len(newX)
    if nReturn > 0:
        print("### Return %d Useful Images" % nReturn)
        # Sort the returned images according to the size of their BBox
        indSize = np.argsort(boxSize)

        # Go through the returned images, put them in the cutout region
        for n in range(nReturn):
            ind = indSize[n]

            # Put in the image array
            imgUse = imgArr[ind]
            imgEmpty[newY[ind]:(newY[ind] + boxY[ind]), newX[ind]:(
                newX[ind] + boxX[ind])] = imgUse[:, :]

            # Put in the mask array
            if not imgOnly:
                mskUse = mskArr[ind]
                mskEmpty[newY[ind]:(newY[ind] + boxY[ind]), newX[ind]:(
                    newX[ind] + boxX[ind])] = mskUse[:, :]
                # Put in the variance array
                varUse = varArr[ind]
                varEmpty[newY[ind]:(newY[ind] + boxY[ind]), newX[ind]:(
                    newX[ind] + boxX[ind])] = varUse[:, :]
                # Convert it into sigma array
                sigEmpty = np.sqrt(varEmpty)

                # Put in the detection mask array
                detUse = detArr[ind]
                detEmpty[newY[ind]:(newY[ind] + boxY[ind]), newX[ind]:(
                    newX[ind] + boxX[ind])] = detUse[:, :]

                # Save the PSF image
                psfOut = outPre + '_psf.fits'
                if ((psfArr[ind] is not None) and
                    (not os.path.isfile(psfOut))):
                    psfUse = psfArr[ind]
                    psfUse.writeFits(psfOut)

            if n == (nReturn - 1):
                # This is the largest available sub-image
                phoZp = zpList[ind]

        # Check if PSF is available
        if not os.path.isfile(psfOut):
            warnings.warn("!!! Can not generate PSF for %s" % prefix)

        # See if all the cutout region is covered by data
        nanPix = np.sum(np.isnan(imgEmpty))
        if nanPix < (sizeExpect * 0.1):
            cutFull = True
        else:
            cutFull = False
            if verbose:
                print("## There are still %d NaN pixels!" % nanPix)
        if not imgOnly:
            # For mask images, replace NaN with a large value: 999
            mskEmpty[np.isnan(mskEmpty)] = 999
            # For detections, replace NaN with 0
            detEmpty[np.isnan(detEmpty)] = 0

        # Create a WCS for the combined image
        outWcs = apWcs.WCS(naxis=2)
        outWcs.wcs.crpix = [newCenX + 1, newCenY + 1]
        outWcs.wcs.crval = [ra, dec]
        outWcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        outWcs.wcs.cdelt = np.array([cdMatrix[0][0], cdMatrix[1][1]])

        # Output to header
        outHead = outWcs.to_header()
        outHead.set("PIXEL", pixScale, "Pixel Scale [arcsec/pix]")
        outHead.set("PHOZP", phoZp, "Photometric Zeropoint")
        outHead.set("EXPTIME", 1.0, "Set exposure time to 1 sec")
        outHead.set("GAIN", 3.0, "Average GAIN for HSC CCDs")
        for m in range(nReturn):
            outHead.set("TRACT" + str(m), trList[m])
            outHead.set("PATCH" + str(m), paList[m])

        # Define the output file name
        if verbose:
            print("\n### Generate Outputs")

        # Save the image array
        saveImageArr(imgEmpty, outHead, outPre + '_img.fits')
        if not imgOnly:
            # Save the mask array
            saveImageArr(mskEmpty, outHead, outPre + '_bad.fits')
            # Save the sigma array
            saveImageArr(sigEmpty, outHead, outPre + '_sig.fits')
            # Save the detection mask array
            saveImageArr(detEmpty, outHead, outPre + '_det.fits')

        # If necessary, save the source catalog
        if ((not imgOnly) and saveSrc and srcFound):
            srcUse = flatSrcArr(srcArr)
            refUse = flatSrcArr(refArr)
            forceUse = flatSrcArr(forceArr)
            # Write out the catalogs
            srcUse.writeFits(outPre + '_meas.fits')
            refUse.writeFits(outPre + '_ref.fits')
            forceUse.writeFits(outPre + '_forced.fits')

        if nReturn > 0:
            cutFound = True
            # Save a preview image
            if visual:
                pngOut = outPre + '_pre.png'
                if not imgOnly:
                    previewCoaddImage(
                        imgEmpty,
                        mskEmpty,
                        varEmpty,
                        detEmpty,
                        oriX=newX,
                        oriY=newY,
                        boxW=boxX,
                        boxH=boxY,
                        outPNG=pngOut)
        else:
            cutFound = False
            print(WAR)
            print("\n!!! No data was collected for " +
                  "this RA,DEC in %s band!" % filt)
    else:
        print(WAR)
        print("\n### No data was collected for this RA,DEC in %s band!" % filt)
        cutFound = False
        cutFull = False

    return cutFound, cutFull, nReturn


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("ra", type=float, help="RA  to search")
    parser.add_argument("dec", type=float, help="Dec to search")
    parser.add_argument("size", type=float, help="Half size of the cutout box")
    parser.add_argument(
        '-f', '--filter', dest='filt', help="Filter", default='HSC-I')
    parser.add_argument(
        '-p',
        '--prefix',
        dest='outfile',
        help='Prefix of the output file',
        default='hsc_coadd_cutout')
    parser.add_argument(
        '-i', '--imgOnly', action="store_true",
        dest='imgOnly', default=False)
    parser.add_argument(
        '-b', '--noBrightStar', action="store_true",
        dest='no_bright_object', default=False)
    args = parser.parse_args()

    coaddImageCutFull(
        args.root,
        args.ra,
        args.dec,
        args.size,
        filt=args.filt,
        prefix=args.outfile,
        imgOnly=args.imgOnly,
        no_bright_object=args.no_bright_object)
