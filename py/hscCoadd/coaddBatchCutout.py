#!/usr/bin/env python
"""Generate HSC full cutout in batch mode."""

import os
import fcntl
import numpy
import argparse
import warnings

from astropy.io import fits
# HSC Pipeline
import lsst.daf.persistence as dafPersist

from coaddImageCutout import coaddImageCutFull, coaddImageCutout
from coaddColourImage import coaddColourImageFull, coaddColourImage

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100
HSC_FILTERS = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']

# For multiprocessing
try:
    from joblib import Parallel, delayed
    multiJob = True
except ImportError:
    multiJob = False


def decideCutoutSize(z, safe=False):
    """
    Decide the typical cutout size for certain redshift.

    Parameters:
        safe  : True will make the cutout smaller
    """
    if (z <= 0.15):
        if safe:
            return 1000
        else:
            return 1200
    elif (z > 0.15) and (z <= 0.25):
        if safe:
            return 650
        else:
            return 750
    elif (z > 0.25) and (z <= 0.35):
        if safe:
            return 550
        else:
            return 600
    elif (z > 0.35) and (z <= 0.45):
        if safe:
            return 350
        else:
            return 400
    elif (z > 0.45) and (z <= 0.65):
        if safe:
            return 300
        else:
            return 350
    elif (z >= 0.65):
        if safe:
            return 200
        else:
            return 250
    else:
        return 300


def parseInputCatalog(input, sizeDefault=300, idField='index',
                      raField='ra', decField='dec', sizeField='size',
                      zField=None, zCutoutSize=False, infoField1=None,
                      infoField2=None, safe=False):
    """
    Parse the input catalog.

    Parameters:
    """
    # Read in the catalog
    cat = fits.open(input)[1].data

    # Try to get the ID, Ra, Dec
    try:
        index = cat.field(idField)
        nObjs = len(index)
    except KeyError:
        raise Exception('Can not find the ID field')

    try:
        ra = cat.field(raField)
    except KeyError:
        raise Exception('Can not find the RA field')

    try:
        dec = cat.field(decField)
    except KeyError:
        raise Exception('Can not find the DEC field')

    if zField is not None:
        try:
            redshift = cat.field(zField)
        except KeyError:
            raise Exception('Can not find the REDSHIFT field')
    else:
        redshift = None

    # Extra information 1
    if infoField1 is not None:
        try:
            info2 = cat.field(infoField1)
        except KeyError:
            print WAR
            print "### Can not find field: %s in the catalog !" % infoField1
            info2 = None
        else:
            if isinstance(info2[0], (float, int, numpy.number)):
                info2 = map(lambda x: "{:10.3f}".format(x).strip(), info2)
    else:
        info2 = None

    # Extra information 2
    if infoField2 is not None:
        try:
            info3 = cat.field(infoField2)
        except KeyError:
            print WAR
            print "### Can not find field: %s in the catalog !" % infoField2
            info3 = None
        else:
            if isinstance(info3[0], (float, int, numpy.number)):
                info3 = map(lambda x: "{:10.3f}".format(x).strip(), info3)
    else:
        info3 = None

    if zCutoutSize and (redshift is not None):
        size = map(lambda x: decideCutoutSize(x, safe=safe), redshift)
        size = numpy.asarray(size)
    else:
        if sizeField is not None:
            try:
                size = cat.field(sizeField)
            except KeyError:
                size = numpy.empty(nObjs)
                size.fill(sizeDefault)
        else:
            size = numpy.empty(nObjs)
            size.fill(sizeDefault)

    return (index, ra, dec, size, redshift, info2, info3), nObjs


def coaddBatchCutout(root, inCat, size=100, band='HSC-I',
                     prefix='coadd_cutout', sample=None, idField='index',
                     raField='ra', decField='dec', colorFilters='gri',
                     sizeField='size', zCutoutSize=False,
                     zField=None, verbose=True, noColor=False,
                     onlyColor=False, infoField1=None, infoField2=None,
                     clean=False, img_min=-0.0, img_max=0.72,
                     Q=15, stitch=False,
                     noName=False):
    """
    Generate HSC coadd cutout images in batch mode.

    Also have the option to generate (or just generate) a 3-band
    color image
    """
    if not os.path.isdir(root):
        raise Exception("Wrong root directory for data! %s" % root)

    if os.path.exists(inCat):
        result = parseInputCatalog(inCat, sizeDefault=size, idField=idField,
                                   raField=raField, decField=decField,
                                   zField=zField, zCutoutSize=zCutoutSize,
                                   infoField1=infoField1,
                                   infoField2=infoField2)
        index, ra, dec, size, z, extr1, extr2 = result
    else:
        raise Exception("### Can not find the input catalog: %s" % inCat)

    if not onlyColor:
        if sample is not None:
            logPre = prefix + '_' + sample
        else:
            logPre = prefix
        logFile = logPre + '_match_' + band.strip() + '.lis'
        if not os.path.isfile(logFile):
            os.system('touch ' + logFile)

    nObjs = len(index)
    if verbose:
        print SEP
        print "### Will try to get cutout image for %d objects" % nObjs
        print SEP

    for i in range(nObjs):

        if verbose:
            print "### %d -- ID: %s ; " % (i+1, str(index[i])) + \
                  "RA: %10.5f DEC %10.5f ; Size: %d" % (ra[i], dec[i], size[i])

        # New prefix
        newPrefix = prefix + '_' + str(index[i]).strip()

        # Cutout Image
        if not onlyColor:
            tempOut = coaddImageCutout(root, ra[i], dec[i], size[i],
                                       saveMsk=True, filt=band,
                                       prefix=newPrefix)
            coaddFound, noData, partialCut = tempOut
            if coaddFound:
                if not noData:
                    if not partialCut:
                        matchStatus = 'Full'
                    else:
                        matchStatus = 'Part'
                else:
                    matchStatus = 'NoData'
            else:
                matchStatus = 'Outside'

            with open(logFile, "a") as logMatch:
                try:
                    logFormat = "%5d    %s    %s \n"
                    logMatch.write(logFormat % (index[i], band,
                                                matchStatus))
                    fcntl.flock(logMatch, fcntl.LOCK_UN)
                except IOError:
                    pass

        # Color Image
        # Whether put redshift on the image
        if (zField is not None) and (z is not None):
            info1 = "z=%5.3f" % z[i]
        else:
            info1 = None
        # Extra information
        if (infoField1 is not None) and (extr1 is not None):
            info2 = str(extr1[i]).strip()
        else:
            info2 = None
        if (infoField2 is not None) and (extr2 is not None):
            info3 = str(extr2[i]).strip()
        else:
            info3 = None

        if onlyColor:
            if noName:
                name = None
            else:
                name = str(index[i])
            if stitch:
                if not clean:
                    coaddColourImageFull(root, ra[i], dec[i], size[i],
                                         filt=colorFilters,
                                         scaleBar=10,
                                         prefix=newPrefix, name=name,
                                         info1=info1, info2=info2,
                                         info3=info3,
                                         img_min=img_min, img_max=img_max,
                                         Q=Q)
                else:
                    coaddColourImageFull(root, ra[i], dec[i], size[i],
                                         filt=colorFilters,
                                         scaleBar=None,
                                         prefix=newPrefix, name=None,
                                         info1=None, info2=None,
                                         info3=None,
                                         img_min=img_min, img_max=img_max,
                                         Q=Q)
            else:
                coaddColourImage(root, ra[i], dec[i], size[i],
                                 filt=colorFilters,
                                 prefix=newPrefix, name=name,
                                 info1=info1, info2=info2, info3=info3,
                                 img_min=img_min, img_max=img_max, Q=Q)
        elif (matchStatus is 'Full') or (matchStatus is 'Part'):
            if noName:
                name = None
            else:
                name = str(index[i])
            if stitch:
                coaddColourImageFull(root, ra[i], dec[i], size[i],
                                     filt=colorFilters,
                                     prefix=newPrefix, name=name,
                                     info1=info1, info2=info2,
                                     info3=info3,
                                     img_min=img_min, img_max=img_max,
                                     Q=Q)
            else:
                coaddColourImage(root, ra[i], dec[i], size[i],
                                 filt=colorFilters,
                                 prefix=newPrefix, name=name,
                                 info1=info1, info2=info2, info3=info3,
                                 img_min=img_min, img_max=img_max, Q=Q)


def singleCut(index, butler, root, useful, config):
    """Make cutout for single object."""
    index, ra, dec, size, z, extr1, extr2 = useful
    band = config['band']
    prefix = config['prefix']
    sample = config['sample']
    colorFilters = config['colorFilters']
    zField = config['zField']
    sizeField = config['sizeField']
    scaleBar = config['scaleBar']
    verbose = config['verbose']
    noColor = config['noColor']
    onlyColor = config['onlyColor']
    infoField1 = config['infoField1']
    infoField2 = config['infoField2']
    clean = config['clean']
    img_min = config['img_min']
    img_max = config['img_max']
    Q = config['Q']
    saveSrc = config['saveSrc']
    makeDir = config['makeDir']
    noName = config['noName']
    imgOnly = config['imgOnly']
    allFilters = config['allFilters']

    if verbose:
        print "### %d -- ID: %s ; " % ((index + 1),
                                       str(index[index])) + \
              "RA: %10.5f DEC %10.5f ; Size: %d" % (ra[index],
                                                    dec[index],
                                                    size[index])
    # New prefix
    newPrefix = prefix + '_' + str(index[index]).strip()
    # Cutout Image
    if not onlyColor:
        if verbose:
            print "### Make the Cutout Fits Files!  "
        if not allFilters:
            filterUse = band.strip()

            if not onlyColor:
                if sample is not None:
                    logPre = prefix + '_' + sample
                else:
                    logPre = prefix
                logFile = logPre + '_match_' + filterUse + '.lis'
                if not os.path.isfile(logFile):
                    os.system('touch ' + logFile)

            if makeDir:
                dirLoc = (str(index[index]).strip() + '/' +
                          str(filterUse).strip() + '/')
                if not os.path.exists(dirLoc):
                    os.makedirs(dirLoc)
                filterPre = dirLoc + newPrefix
            else:
                filterPre = newPrefix

            if saveSrc:
                tempOut = coaddImageCutFull(root,
                                            ra[index],
                                            dec[index],
                                            size[index],
                                            savePsf=True,
                                            saveSrc=True,
                                            visual=True,
                                            filt=filterUse,
                                            prefix=filterPre,
                                            butler=butler,
                                            imgOnly=imgOnly)
                found, full, npatch = tempOut
            else:
                tempOut = coaddImageCutFull(root,
                                            ra[index],
                                            dec[index],
                                            size[index],
                                            savePsf=True,
                                            saveSrc=False,
                                            visual=True,
                                            filt=filterUse,
                                            prefix=filterPre,
                                            butler=butler,
                                            imgOnly=imgOnly)
                found, full, npatch = tempOut
            if found:
                matchStatus = 'Found'
                if full:
                    full = 'Full'
                else:
                    full = 'Part'
            else:
                matchStatus = 'NoData'
                full = 'None'

            with open(logFile, "a") as logMatch:
                logStr = "%10s   %s   %6s   %4s   %3d \n"
                try:
                    logMatch.write(logStr % (str(index[index]),
                                             filterUse,
                                             matchStatus,
                                             full, npatch))
                    fcntl.flock(logMatch, fcntl.LOCK_UN)
                except IOError:
                    pass
        else:
            for filterUse in HSC_FILTERS:
                print "## Working on %s now" % filterUse

                if not onlyColor:
                    if sample is not None:
                        logPre = prefix + '_' + sample
                    else:
                        logPre = prefix
                    logFilter = logPre + '_match_' + filterUse + '.lis'
                    if not os.path.isfile(logFilter):
                        os.system('touch ' + logFilter)

                if makeDir:
                    dirLoc = (str(index[index]).strip() + '/' +
                              str(filterUse).strip() + '/')
                    if not os.path.exists(dirLoc):
                        os.makedirs(dirLoc)
                    filterPre = dirLoc + newPrefix
                else:
                    filterPre = newPrefix

                if saveSrc:
                    tempOut = coaddImageCutFull(root,
                                                ra[index],
                                                dec[index],
                                                size[index],
                                                savePsf=True,
                                                saveSrc=True,
                                                visual=True,
                                                filt=filterUse,
                                                prefix=filterPre,
                                                butler=butler,
                                                imgOnly=imgOnly)
                    found, full, npatch = tempOut
                else:
                    tempOut = coaddImageCutFull(root,
                                                ra[index],
                                                dec[index],
                                                size[index],
                                                savePsf=True,
                                                saveSrc=False,
                                                visual=True,
                                                filt=filterUse,
                                                prefix=filterPre,
                                                butler=butler,
                                                imgOnly=imgOnly)
                    found, full, npatch = tempOut
                if found:
                    matchStatus = 'Found'
                    if full:
                        full = 'Full'
                    else:
                        full = 'Part'
                else:
                    matchStatus = 'NoData'
                    full = 'None'

                with open(logFilter, "a") as logMatch:
                    logStr = "%5d   %s   %6s   %4s   %3d \n"
                    try:
                        logMatch.write(logStr % (index[index],
                                                 filterUse,
                                                 matchStatus,
                                                 full, npatch))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass

    # Color Image
    # Whether put redshift on the image
    if (zField is not None) and (z is not None):
        info1 = "z=%5.3f" % z[index]
    else:
        info1 = None
    # Extra information
    if (infoField1 is not None) and (extr1 is not None):
        info2 = str(extr1[index]).strip()
    else:
        info2 = None
    if (infoField2 is not None) and (extr2 is not None):
        info3 = str(extr2[index]).strip()
    else:
        info3 = None

    if onlyColor:
        if noName:
            name = None
        else:
            name = str(index[index])
        if verbose:
            print "### Generate Color Image !"
        if clean:
            coaddColourImageFull(root,
                                 ra[index], dec[index],
                                 size[index],
                                 filt=colorFilters,
                                 prefix=newPrefix, name=None,
                                 info1=None, info2=None,
                                 info3=None,
                                 scaleBar=None,
                                 img_min=img_min, img_max=img_max, Q=Q,
                                 butler=butler)
        else:
            coaddColourImageFull(root,
                                 ra[index], dec[index],
                                 size[index],
                                 filt=colorFilters,
                                 prefix=newPrefix, name=name,
                                 info1=info1, info2=info2,
                                 info3=info3,
                                 scaleBar=scaleBar,
                                 img_min=img_min, img_max=img_max, Q=Q,
                                 butler=butler)
    elif (matchStatus is 'Found' and not noColor):
        if noName:
            name = None
        else:
            name = str(index[index])
        if verbose:
            print "### Generate Color Image !"
        coaddColourImageFull(root, ra[index],
                             dec[index], size[index],
                             filt=colorFilters,
                             prefix=newPrefix, name=name,
                             info1=info1, info2=info2, info3=info3,
                             img_min=img_min, img_max=img_max,
                             Q=Q, butler=butler)


def coaddBatchCutFull(root, inCat, size=100, band='HSC-I',
                      prefix='coadd_cutout', sample=None, idField='index',
                      raField='ra', decField='dec', colorFilters='gri',
                      sizeField='size', zCutoutSize=False, zField=None,
                      verbose=True, noColor=False, onlyColor=False,
                      infoField1=None, infoField2=None, clean=False,
                      img_min=-0.0, img_max=0.72, Q=15, safe=False, saveSrc=False,
                      makeDir=False, noName=False, njobs=1,
                      imgOnly=False, allFilters=False, scaleBar=10.0):
    """
    Generate HSC coadd cutout images.

    Also have the option to generate (or just generate) a 3-band
    color image
    """
    if not os.path.isdir(root):
        raise Exception("Wrong root directory for data! %s" % root)
    else:
        try:
            butler = dafPersist.Butler(root)
            if verbose:
                "### Load in the Butler "
        except Exception:
            warnings.warn("### Can not load the Butler!")

    if os.path.exists(inCat):
        if verbose:
            print COM
            print "              PARSE THE INPUT CATALOG                 "
            print COM
        useful, nObjs = parseInputCatalog(inCat, sizeDefault=size,
                                          idField=idField,
                                          raField=raField,
                                          decField=decField,
                                          zField=zField,
                                          sizeField=sizeField,
                                          zCutoutSize=zCutoutSize,
                                          infoField1=infoField1,
                                          infoField2=infoField2,
                                          safe=safe)
    else:
        raise Exception("### Can not find the input catalog: %s" % inCat)

    if verbose:
        print SEP
        print "### Will try to get cutout image for %d objects" % nObjs
        print SEP
    indexObj = numpy.asarray(range(nObjs))

    config = {'band': band,
              'prefix': prefix,
              'sample': sample,
              'colorFilters': colorFilters,
              'zField': zField,
              'sizeField': sizeField,
              'verbose': verbose,
              'noColor': noColor,
              'onlyColor': onlyColor,
              'infoField1': infoField1,
              'infoField2': infoField2,
              'clean': clean,
              'img_min': img_min,
              'img_max': img_max,
              'Q': Q,
              'saveSrc': saveSrc,
              'makeDir': makeDir,
              'noName': noName,
              'imgOnly': imgOnly,
              'allFilters': allFilters,
              'scaleBar': scaleBar
              }

    if njobs > 1 and multiJob:
        """Start parallel run."""
        Parallel(n_jobs=njobs)(delayed(singleCut)(
                               index, butler, root, useful,
                               config) for index in indexObj)
    else:
        for index in indexObj:
            singleCut(index, butler, root, useful, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("incat", help="The input catalog for cutout")
    parser.add_argument("-s", '--size', dest='size', type=int,
                        help="Half size of the cutout box", default=200)
    parser.add_argument('-f', '--filter', dest='filt', help="Filter",
                        default='HSC-I')
    parser.add_argument('--sample', dest='sample', help="Sample name",
                        default=None)
    parser.add_argument('-j', '--njobs', type=int,
                        help='Number of jobs run at the same time',
                        dest='njobs', default=1)
    parser.add_argument('-p', '--prefix', dest='prefix',
                        help='Prefix of the output file',
                        default='hsc_coadd_cutout')
    parser.add_argument('-id', '--id', dest='idField',
                        help="Column name for ID", default='index')
    parser.add_argument('-ra', '--ra', dest='raField',
                        help="Column name for RA",
                        default='ra_hsc')
    parser.add_argument('-dec', '--dec', dest='decField',
                        help="Column name for DEC",
                        default='dec_hsc')
    parser.add_argument('-z', '--redshift', dest='zField',
                        help="Column name for z",
                        default='z_use')
    parser.add_argument('-cf', '--color-filters', dest='colorFilters',
                        help="Choice of filters for color images",
                        default='riz')
    parser.add_argument('-sf', '--size-field', dest='sizeField',
                        help="Column name for cutout size",
                        default=None)
    parser.add_argument('-info1', '--infoField1', dest='infoField1',
                        help="Column name for first extra information",
                        default=None)
    parser.add_argument('-info2', '--infoField2', dest='infoField2',
                        help="Column name for second extra information",
                        default=None)
    parser.add_argument('-af', '--allFilters', action="store_true",
                        dest='allFilters', default=False)
    parser.add_argument('-img', '--imgOnly', action="store_true",
                        dest='imgOnly', default=False)
    parser.add_argument('-zc', '--zCutoutSize', action="store_true",
                        dest='zCutoutSize', default=True)
    parser.add_argument('-nc', '--noColor', action="store_true",
                        dest='noColor', default=True)
    parser.add_argument('-oc', '--onlyColor', action="store_true",
                        dest='onlyColor', default=False)
    parser.add_argument('-safe', '--safe', action="store_true", dest='safe',
                        default=False)
    parser.add_argument('-clean', '--clean', action="store_true", dest='clean',
                        default=False)
    parser.add_argument('-nn', '--noName', action="store_true",
                        dest='noName', default=False)
    parser.add_argument('-v', '--verbose', action="store_true", dest='verbose',
                        default=False)
    parser.add_argument('-src', '--src', action="store_true", dest='saveSrc',
                        default=False)
    parser.add_argument('-makeDir', '--makeDir', action="store_true",
                        dest='makeDir', default=False)
    parser.add_argument('-b', '--scalebar', type=float,
                        help='Size of the scale bar in unit of arcsec',
                        dest='scaleBar', default=10.0)
    args = parser.parse_args()

    coaddBatchCutFull(args.root, args.incat, size=args.size,
                      band=args.filt, prefix=args.prefix,
                      idField=args.idField, raField=args.raField,
                      decField=args.decField, sizeField=args.sizeField,
                      colorFilters=args.colorFilters, zField=args.zField,
                      zCutoutSize=args.zCutoutSize, noColor=args.noColor,
                      onlyColor=args.onlyColor, infoField1=args.infoField1,
                      infoField2=args.infoField2, safe=args.safe,
                      verbose=args.verbose, clean=args.clean,
                      saveSrc=args.saveSrc, sample=args.sample,
                      makeDir=args.makeDir, noName=args.noName,
                      imgOnly=args.imgOnly, allFilters=args.allFilters,
                      njobs=args.njobs, scaleBar=args.scaleBar)
