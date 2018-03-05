#!/usr/bin/env python
"""Generate color picture of HSC cutout."""

from __future__ import (division, print_function)

import argparse
import numpy as np

import lsst.daf.persistence as dafPersist
import lsst.afw.display.rgb as afwRgb
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def getCircleRaDec(ra, dec, size):
    """Get a set of (RA, DEC) that describe a circle."""
    # Convert the size from pixel unit to degress
    sizeDegree = (size * 0.168) / 3600.0
    # representative set of polar angles
    angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    phi = np.array(angles * np.pi / 180.0)

    # Get the (Ra, Dec) coordinates of these points
    raList = ra + sizeDegree * np.cos(phi)
    decList = dec + sizeDegree * np.sin(phi)

    # Also include the center
    raList = np.append(raList, ra)
    decList = np.append(decList, dec)

    return raList, decList


def saveRgbPng(outRgb,
               imgRgb,
               cenMark=False,
               xCen=None,
               yCen=None,
               name=None,
               info1=None,
               info2=None,
               info3=None,
               sLength=None,
               sString=None):
    """Save the RGB image as a PNG figure."""
    # Decide the image size
    sizeX, sizeY, dim = imgRgb.shape
    sizeX = int(sizeX / 100) if (sizeX / 100) < 15 else 15
    sizeY = int(sizeY / 100) if (sizeY / 100) < 15 else 15
    sizeX = sizeX if sizeX > 6 else 6
    sizeY = sizeY if sizeY > 6 else 6

    fig = plt.figure(figsize=(sizeX, sizeY), dpi=100, frameon=False)

    # Show the image
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(imgRgb, interpolation='none', origin='lower')
    ax.set_aspect('equal')

    # Suppress all the ticks and tick labels
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # Highlight the new center: TODO
    if cenMark and (xCen is not None) and (yCen is not None):
        ax.scatter(
            xCen,
            yCen,
            s=80,
            lw=0.5,
            marker='o',
            edgecolors='r',
            facecolors='none')

    # Add some information on the image
    if name is not None:
        ax.text(
            0.5,
            0.10,
            name,
            fontsize=20,
            fontweight='bold',
            ha='center',
            va='center',
            color='w',
            transform=ax.transAxes)
    if info1 is not None:
        ax.text(
            0.8,
            0.90,
            info1,
            fontsize=18,
            fontweight='bold',
            ha='center',
            va='center',
            color='w',
            transform=ax.transAxes)
    if info2 is not None:
        ax.text(
            0.8,
            0.82,
            info2,
            fontsize=18,
            fontweight='bold',
            ha='center',
            va='center',
            color='w',
            transform=ax.transAxes)
    if info3 is not None:
        ax.text(
            0.8,
            0.74,
            info3,
            fontsize=18,
            fontweight='bold',
            ha='center',
            va='center',
            color='w',
            transform=ax.transAxes)
    if sLength is not None:
        ax.plot(
            [0.14, 0.14 + sLength], [0.88, 0.88],
            'w-',
            lw=2.5,
            transform=ax.transAxes)
        if sString is not None:
            ax.text(
                (0.28 + sLength) / 2.0,
                0.85,
                sString,
                fontsize=15,
                ha='center',
                va='center',
                color='w',
                fontweight='bold',
                transform=ax.transAxes)

    ax.margins(0.00, 0.00, tight=True)

    fig.savefig(outRgb, bbox_inches='tight', pad_inches=0, ad_inches=0)
    plt.close(fig)


def isHscFilter(filter, short=True):
    """Check if this is the right filter."""
    if not short:
        hscFilters = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
    else:
        hscFilters = ['g', 'r', 'i', 'z', 'y']

    return (filter in hscFilters)


def coaddColourImage(root,
                     ra,
                     dec,
                     size,
                     filt='gri',
                     prefix='hsc_coadd_cutout',
                     info1=None,
                     info2=None,
                     info3=None,
                     min=-0.0,
                     max=0.70,
                     Q=10,
                     name=None,
                     localMax=True,
                     scaleBar=10,
                     butler=None):
    """Generate colored picture for cutout."""
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
            print('\n### Can not load the correct Butler!')
    skyMap = butler.get("deepCoadd_skyMap", immediate=True)

    # [Ra, Dec] pair
    raDec = afwCoord.Coord(ra * afwGeom.degrees, dec * afwGeom.degrees)
    # Expected size and center position
    dimExpect = np.int(2 * size + 1)
    cenExpect = (dimExpect / 2.0, dimExpect / 2.0)
    # Create a empty array
    # For RGB image, the data type should be uint8
    rgbEmpty = np.zeros((dimExpect, dimExpect, 3), dtype="uint8")

    # Check the choice of filters
    if len(filt) is not 3:
        raise Exception("Have to be three filters!")
    elif not (isHscFilter(filt[0]) & isHscFilter(filt[1]) &
              isHscFilter(filt[2])):
        raise Exception("Not all filters are valid !")
    """
    Figure out the area we want, and read the data.
    For coadds the WCS is the same in all bands, but the code handles
    the general case. Start by finding the tract and patch
    """
    tractInfo, patchInfo = skyMap.findTractPatchList([raDec])[0]
    tract = tractInfo.getId()
    patch = "%d,%d" % patchInfo[0].getIndex()

    # Check if the coordinate is available in all three bands.
    try:
        # Get the correct HSC filter name
        filter1 = "HSC-%s" % filt[0].upper()
        filter2 = "HSC-%s" % filt[1].upper()
        filter3 = "HSC-%s" % filt[2].upper()
        filtArr = [filter1, filter2, filter3]
    except Exception, errMsg:
        print("\n !!! The desired coordinate is not available !!! ")
        print(errMsg)
    else:
        # Then we can read the desired pixels
        images = {}
        # newcen = {}
        newX = {}
        newY = {}
        cutoutSize = int(size)

        for i in range(3):
            # Find the file of the coadd image
            coadd = butler.get(
                coaddData, tract=tract, patch=patch, filter=filtArr[i])
            # Get the WCS information
            wcs = coadd.getWcs()
            # Convert the central coordinate from Ra,Dec to pixel unit
            pixel = wcs.skyToPixel(raDec)
            pixel = afwGeom.Point2I(pixel)
            # Define the bounding box for the central pixel
            bbox = afwGeom.Box2I(pixel, pixel)
            # Grow the bounding box to the desired size
            bbox.grow(int(cutoutSize))
            xOri, yOri = bbox.getBegin()

            # Compare to the coadd image, and clip
            bbox.clip(coadd.getBBox(afwImage.PARENT))
            newX[i] = bbox.getBeginX() - xOri
            newY[i] = bbox.getBeginY() - yOri
            # Get the masked image
            subImage = afwImage.ExposureF(coadd, bbox, afwImage.PARENT)
            # Get the WCS
            # subWcs = subImage.getWcs()
            # Get the central pixel coordinates on new subImage WCS
            # newcen[i] = subWcs.skyToPixel(raDec)
            # Extract the image array
            images[i] = subImage.getMaskedImage().getImage()

        # Define the Blue, Green, and Red channels
        # These cutouts are still HSC ImageF object, not numpy array
        bCut, gCut, rCut = images[0], images[1], images[2]

        if localMax:
            maxArr = []
            for m in range(3):
                imgPad = np.zeros(
                    (int(dimExpect), int(dimExpect)), dtype=float)
                imgPad[newY[m]:(newY[m] + images[m].getHeight()), newX[m]:(
                    newX[m] + images[m].getWidth())] = images[m].getArray()
                globalMax = np.max(images[m].getArray())
                localMax = np.max(imgPad[cenExpect[0] - 10:cenExpect[0] + 10,
                                         cenExpect[1] - 10:cenExpect[1] + 10])
                maxArr.append(localMax / globalMax)
            maxShow = np.max(np.asarray(maxArr))
        else:
            maxShow = max
        # To see if data are available for all the cut-out region
        if (bCut.getHeight() < dimExpect) or (bCut.getWidth() < dimExpect):
            print("\n### Only part of the desired cutout-region is returned !")
            # Define the name of the output file
            outRgb = prefix + '_' + filt + '_part_color.png'
            partial = True
        else:
            outRgb = prefix + '_' + filt + '_color.png'
            partial = False
        # Generate the RGB image
        # 15/04/22: min ==> minimum
        imgRgb = afwRgb.makeRGB(
            rCut,
            gCut,
            bCut,
            minimum=min,
            dataRange=(maxShow - min),
            Q=Q,
            saturatedPixelValue=None)
        if partial:
            for k in range(3):
                rgbEmpty[newY[k]:(newY[k] + images[k].getHeight()), newX[k]:(
                    newX[k] + images[k].getWidth()), k] = imgRgb[:, :, k]
            imgRgb = rgbEmpty
        # Add a scale bar
        if scaleBar is not None:
            sLength = ((scaleBar * 1.0) / 0.168) / (dimExpect * 1.0)
            sString = "%d\"" % int(scaleBar)
        else:
            sLength = None
            sString = None
        # Better way to show the image
        if partial:
            saveRgbPng(
                outRgb,
                imgRgb,
                cenMark=True,
                xCen=cenExpect[0],
                yCen=cenExpect[1],
                sLength=sLength,
                sString=sString,
                name=name,
                info1=info1,
                info2=info2,
                info3=info3)
        else:
            saveRgbPng(
                outRgb,
                imgRgb,
                name=name,
                info1=info1,
                info2=info2,
                info3=info3,
                sLength=sLength,
                sString=sString)


def getTractPatchList(matches):
    """Get the list of Tract, Patch."""
    tract = []
    patch = []

    for match in matches:
        tractInfo, patchInfo = match
        tractId = tractInfo.getId()
        for patchItem in patchInfo:
            tract.append(tractId)
            patch.append("%d,%d" % patchItem.getIndex())

    return tract, patch


def getFitsImgName(root, tract, patch, filter, imgType='deepCoadd_calexp'):
    """Get the fits name."""
    if root[-1] is not '/':
        root += '/'
    imgName = (root + imgType + '/' + filter + '/' + str(tract) + '/' + patch +
               '.fits')

    return imgName


def coaddColourImageFull(root,
                         ra,
                         dec,
                         size,
                         filt='gri',
                         prefix='hsc_coadd_cutout',
                         info1=None,
                         info2=None,
                         info3=None,
                         min=-0.0,
                         max=0.70,
                         Q=10,
                         name=None,
                         localMax=True,
                         scaleBar=10,
                         butler=None,
                         verbose=False):
    """General full colored picture of cutout."""
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
            print('\n### Can not load the correct Butler!')

    skyMap = butler.get("deepCoadd_skyMap", immediate=True)

    # [Ra, Dec] list
    raDec = afwCoord.Coord(ra * afwGeom.degrees, dec * afwGeom.degrees)
    raList, decList = getCircleRaDec(ra, dec, size)
    points = map(lambda x, y: afwGeom.Point2D(x, y), raList, decList)
    raDecList = map(lambda x: afwCoord.IcrsCoord(x), points)

    # Expected size and center position
    dimExpect = int(2 * size + 1)
    # cenExpect = (dimExpect / 2.0, dimExpect / 2.0)
    # Create a empty array
    # For RGB image, the data type should be uint8
    rgbEmpty = np.zeros((dimExpect, dimExpect, 3), dtype="uint8")

    # Check the choice of filters
    if len(filt) is not 3:
        raise Exception("Have to be three filters!")
    elif not (isHscFilter(filt[0]) & isHscFilter(filt[1]) &
              isHscFilter(filt[2])):
        raise Exception("Not all filters are valid !")

    # Get the correct HSC filter name
    filter1 = "HSC-%s" % filt[0].upper()
    filter2 = "HSC-%s" % filt[1].upper()
    filter3 = "HSC-%s" % filt[2].upper()
    filtArr = [filter1, filter2, filter3]
    # Cutout size
    cutoutSize = int(size)

    """
    Figure out the area we want, and read the data.
    For coadds the WCS is the same in all bands,
    but the code handles the general case
    Start by finding the tract and patch
    """
    matches = skyMap.findTractPatchList(raDecList)
    tractList, patchList = getTractPatchList(matches)
    nPatch = len(patchList)

    # Output RGB image
    if verbose:
        print("\n### WILL DEAL WITH %d (TRACT, PATCH)" % nPatch)

    outRgb = prefix + '_' + filt + '_color.png'

    newX = []
    newY = []
    boxX = []
    boxY = []
    boxSize = []
    rgbArr = []
    # Go through all these images
    for j in range(nPatch):
        # Tract, patch
        tract, patch = tractList[j], patchList[j]
        if verbose:
            print("\n### Dealing with %d - %s" % (tract, patch))
        # Check if the coordinate is available in all three bands.
        # Change the method, try to generate something as long as it is
        # covered by at least one band
        images = {}
        for i in range(3):
            try:
                # Find the coadd image
                coadd = butler.get(
                    coaddData,
                    tract=tract,
                    patch=patch,
                    filter=filtArr[i],
                    immediate=True)
                # Get the WCS information
                wcs = coadd.getWcs()
                # Convert the central coordinate from Ra,Dec to pixel unit
                pixel = wcs.skyToPixel(raDec)
                pixel = afwGeom.Point2I(pixel)
                # Define the bounding box for the central pixel
                bbox = afwGeom.Box2I(pixel, pixel)
                # Grow the bounding box to the desired size
                bbox.grow(int(cutoutSize))
                xOri, yOri = bbox.getBegin()
                # Compare to the coadd image, and clip
                bbox.clip(coadd.getBBox(afwImage.PARENT))

                subImage = afwImage.ExposureF(coadd, bbox, afwImage.PARENT)
                # Extract the image array
                images[i] = subImage.getMaskedImage().getImage()
                # Get the size and begginng coordinates of the BBox
                bWidth = bbox.getWidth()
                bHeight = bbox.getHeight()
                bXbegin = bbox.getBeginX()
                bYbegin = bbox.getBeginY()
            except Exception:
                print("\n### Not available in %d - %s - %s" % (tract, patch,
                                                               filtArr[i]))
                images[i] = None

        if not ((images[0] is None) and (images[1] is None) and
                (images[2] is None)):
            # Image from at least one band is available
            # So bWidth, bHeight, bXbegin, bYbegin should be defined
            boxX.append(bWidth)
            boxY.append(bHeight)
            boxSize.append(bWidth * bHeight)
            newX.append(bXbegin - xOri)
            newY.append(bYbegin - yOri)
            for l in range(3):
                if images[0] is not None:
                    bCut = images[0]
                else:
                    # Replace the unavailable data with zero array XXX
                    bCut = np.zeros([bHeight, bWidth])
                if images[1] is not None:
                    gCut = images[1]
                else:
                    # Replace the unavailable data with zero array XXX
                    gCut = np.zeros([bHeight, bWidth])
                if images[2] is not None:
                    rCut = images[2]
                else:
                    # Replace the unavailable data with zero array XXX
                    rCut = np.zeros([bHeight, bWidth])
            # Generate the RGB image
            # 15/04/22: min ==> minimum
            imgRgb = afwRgb.makeRGB(
                rCut,
                gCut,
                bCut,
                minimum=min,
                dataRange=(max - min),
                Q=Q,
                saturatedPixelValue=None)
            rgbArr.append(imgRgb)
        else:
            # Bypass the bad data
            print("\n### NO DATA IS AVAILABLE IN %d - %s" % (tract, patch))

    # Number of returned RGB image
    nReturn = len(rgbArr)
    if verbose:
        print("\n### Return %d Useful Images" % nReturn)
    if nReturn > 0:
        if len(rgbArr) != len(boxSize):
            raise Exception("### Something is weird here !")
        indSize = np.argsort(boxSize)
        # Go through the returned images, put them in the cutout region
        for n in range(nReturn):
            ind = indSize[n]
            # This could lead to problem FIXME
            rgbUse = rgbArr[ind]
            for k in range(3):
                rgbEmpty[newY[ind]:(newY[ind] + boxY[ind]), newX[ind]:(
                    newX[ind] + boxX[ind]), k] = rgbUse[:, :, k]

        imgRgb = rgbEmpty
        # Add a scale bar
        if scaleBar is not None:
            sLength = ((scaleBar * 1.0) / 0.168) / (dimExpect * 1.0)
            sString = "%d\"" % int(scaleBar)
        else:
            sLength = None
            sString = None

        # Better way to show the image
        saveRgbPng(
            outRgb,
            imgRgb,
            name=name,
            info1=info1,
            info2=info2,
            info3=info3,
            sLength=sLength,
            sString=sString)
    else:
        print("\n### NO COLOR IMAGE IS GENERATED FOR THIS OBJECT !!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("ra", type=float, help="RA  to search")
    parser.add_argument("dec", type=float, help="Dec to search")
    parser.add_argument("size", type=float, help="Half size of the cutout box")
    parser.add_argument(
        '-f',
        '--filters',
        dest='filt',
        help="Combination of 3 filters for the colour image",
        default='gri')
    parser.add_argument(
        '-p',
        '--prefix',
        dest='outfile',
        help='Prefix of the output file',
        default='hsc_coadd_cutout')
    parser.add_argument(
        '-i',
        '--info',
        dest='info',
        help='Information to show on the image',
        default=None)
    args = parser.parse_args()

    coaddColourImageFull(
        args.root,
        args.ra,
        args.dec,
        args.size,
        filt=args.filt,
        prefix=args.outfile,
        info1=args.info,
        verbose=True)
