#!/usr/bin/env python
# encoding: utf-8
"""Run ELLIPSE in force photometry mode."""

import os
import gc
import glob
import fcntl
import logging
import warnings
import argparse

from astropy.io import fits

import coaddCutoutSbp as cSbp

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def run(args):
    """
    Run coaddCutoutSbp in batch mode.

    Parameters:
    """
    gc.collect()
    if os.path.isfile(args.incat):
        data = fits.open(args.incat)[1].data
        id = (args.id)
        rerun = (args.rerun).strip()
        prefix = (args.prefix).strip()
        suffix = (args.suffix).strip()
        filter = (args.filter).strip().upper()

        """ Keep a log """
        if args.sample is not None:
            logPre = prefix + '_' + args.sample
        else:
            logPre = prefix
        if args.imgSub:
            logFile = logPre + '_force_imgsub_' + filter.strip() + '.log'
        else:
            logFile = logPre + '_force_img_' + filter.strip() + '.log'
        if not os.path.isfile(logFile):
            os.system('touch ' + logFile)

        if args.verbose:
            print "## Will deal with %d galaxies ! " % len(data)

        for galaxy in data:
            """ ID and prefix """
            galID = str(galaxy[id]).strip()
            galPrefix = prefix + '_' + galID + '_' + filter + '_full'
            galRoot = os.path.join(galID, filter)
            if args.verbose:
                print "## Deal with %s now !" % (galID)

            """ Check the directory of the data """
            if not os.path.isdir(galRoot):
                logging.warning('### Can not find ' +
                                'ROOT folder for %s' % galRoot)
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s    %s    NODIR \n"
                        logMatch.write(logFormat % (galPrefix, filter))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
                continue

            """ Check the input image """
            galImg = galPrefix + '_img.fits'
            if (not os.path.isfile(os.path.join(galRoot, galImg)) and not
                    os.path.islink(os.path.join(galRoot, galImg))):
                logging.warning('### Can not find ' +
                                'CUTOUT IMAGE for %s' % galPrefix)
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s    %s    NOIMG \n"
                        logMatch.write(logFormat % (galPrefix, filter))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
                continue

            """
            Set up a rerun
            """
            galRoot = os.path.join(galRoot, rerun.strip())
            if not os.path.isdir(galRoot):
                os.makedirs(galRoot)
            """ Link the necessary files to the rerun folder """
            fitsList = glob.glob(os.path.join(galRoot, '*.fits'))
            for fitsFile in fitsList:
                seg = fitsFile.split('/')
                link = os.path.join(galRoot, seg[-1])
                if (not os.path.islink(link)) and (not os.path.isfile(link)):
                    os.symlink(fitsFile, link)

            """
            External mask
            """
            if args.maskFilter is not None:
                mskFilter = (args.maskFilter).strip().upper()
                if args.verbose:
                    print "###  Use %s filter for mask \n" % mskFilter
                mskPrefix = prefix + '_' + galID + '_' + mskFilter + '_full'
                mskRoot = os.path.join(galID, mskFilter, rerun)
                galMsk = os.path.join(mskRoot, mskPrefix + '_mskfin.fits')
                if not os.path.isfile(galMsk):
                    logging.warning('### Can not find ' +
                                    'MASK for  %s ' % str(id))
                    with open(logFile, "a") as logMatch:
                        try:
                            logFormat = "%25s    %s    NOMSK \n"
                            logMatch.write(logFormat % (galPrefix, filter))
                            fcntl.flock(logMatch, fcntl.LOCK_UN)
                        except IOError:
                            pass
                    continue
            else:
                galMsk = None

            """
            Input Ellip Binary File
            """
            """ The reference filter """
            refFilter = (args.refFilter).strip().upper()
            """ The reference rerun """
            if args.refRerun is not None:
                refRerun = (args.refRerun).strip()
            else:
                refRerun = rerun
            """ Location and Prefix """
            if args.imgSub:
                imgType = 'imgsub'
            else:
                imgType = 'img'

            galRefRoot = os.path.join(galID, refFilter, refRerun)
            galRefPrefix = (prefix + '_' + galID + '_' + refFilter +
                            '_full_' + imgType + '_ellip_' + refRerun + '_')

            """ The reference model """
            inEllipPrefix = os.path.join(galRefRoot, galRefPrefix)
            refModel = (args.refModel).strip()
            inEllipBin = inEllipPrefix + refModel + '.bin'
            if args.verbose:
                print SEP
                print "###   INPUT ELLIP BIN : %s" % inEllipBin
                print SEP
            if not os.path.isfile(inEllipBin):
                logging.warning('### Can not find ' +
                                'INPUT BINARY for : %s' % galPrefix)
                logging.warning('###     File Name : %s' % inEllipBin)
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s  %20s  %s  NELL \n"
                        logMatch.write(logFormat % (galPrefix,
                                                    inEllipPrefix,
                                                    filter))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
                continue

            """
            Suffix of the output file
            """
            if len(refModel) > 1:
                suffix = refModel[:-1]
                if suffix[-1] == '_':
                    suffix = suffix[:-1]
                ellipSuffix = rerun + '_' + suffix
            else:
                ellipSuffix = rerun

            """ Start a Ellipse run """
            try:
                cSbp.coaddCutoutSbp(galPrefix, root=galRoot,
                                    verbose=args.verbose,
                                    psf=args.psf,
                                    inEllip=inEllipBin,
                                    bkgCor=args.bkgCor,
                                    zp=args.zp,
                                    step=4,
                                    galX0=args.galX0,
                                    galY0=args.galY0,
                                    galQ0=args.galQ0,
                                    galPA0=args.galPA0,
                                    galRe=args.galRe,
                                    checkCenter=args.noCheckCenter,
                                    updateIntens=args.updateIntens,
                                    pix=args.pix,
                                    plot=args.plot,
                                    redshift=args.redshift,
                                    olthresh=args.olthresh,
                                    fracBad=args.fracBad,
                                    lowClip=args.lowClip,
                                    uppClip=args.uppClip,
                                    nClip=args.nClip,
                                    intMode=args.intMode,
                                    minIt=args.minIt,
                                    maxIt=args.maxIt,
                                    maxTry=args.maxTry,
                                    outRatio=args.outRatio,
                                    exMask=galMsk,
                                    suffix=ellipSuffix,
                                    plMask=args.plmask,
                                    imgSub=args.imgSub,
                                    isophote=args.isophote,
                                    xttools=args.xttools)
                logging.info('### The 1-D SBP is DONE for %s in %s' %
                             (galPrefix, filter))
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s  %20s  %s  DONE \n"
                        logMatch.write(logFormat % (galPrefix,
                                                    ellipSuffix,
                                                    filter))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
                gc.collect()

                """ Forced photoetry using small """
                if (galMsk is not None) and args.multiMask:
                    mskSmall = galMsk.replace('mskfin', 'msksmall')
                    mskLarge = galMsk.replace('mskfin', 'msklarge')
                    if args.verbose:
                        print SEP
                        print "##  MultiMask Mode "
                        print "##     Input Ellipse : %s" % inEllipBin
                    """ Small Mask """
                    if os.path.isfile(mskSmall):
                        if args.verbose:
                            print "##     Input MaskSmall : %s" % mskSmall
                        suffixSmall = ellipSuffix + '_msksmall'
                        try:
                            cSbp.coaddCutoutSbp(galPrefix, root=galRoot,
                                                verbose=args.verbose,
                                                psf=False,
                                                inEllip=inEllipBin,
                                                bkgCor=args.bkgCor,
                                                zp=args.zp,
                                                step=4,
                                                galX0=args.galX0,
                                                galY0=args.galY0,
                                                galQ0=args.galQ0,
                                                galPA0=args.galPA0,
                                                galRe=args.galRe,
                                                checkCenter=args.noCheckCenter,
                                                updateIntens=args.updateIntens,
                                                pix=args.pix,
                                                plot=args.plot,
                                                redshift=args.redshift,
                                                olthresh=args.olthresh,
                                                fracBad=args.fracBad,
                                                lowClip=args.lowClip,
                                                uppClip=args.uppClip,
                                                nClip=args.nClip,
                                                intMode=args.intMode,
                                                minIt=args.minIt,
                                                maxIt=args.maxIt,
                                                maxTry=args.maxTry,
                                                outRatio=args.outRatio,
                                                exMask=mskSmall,
                                                suffix=suffixSmall,
                                                plMask=args.plmask,
                                                imgSub=args.imgSub,
                                                isophote=args.isophote,
                                                xttools=args.xttools)
                            logging.info('### SMALLMASK is DONE for %s' %
                                         galPrefix)
                            with open(logFile, "a") as logMatch:
                                try:
                                    logFormat = "%25s  %20s  %s  DONE \n"
                                    logMatch.write(logFormat % (galPrefix,
                                                                suffixSmall,
                                                                filter))
                                    fcntl.flock(logMatch, fcntl.LOCK_UN)
                                except IOError:
                                    pass

                            gc.collect()
                        except Exception, errMsg:
                            print WAR
                            print str(errMsg)
                            print WAR
                            logging.warning('### SMALLMASK FAILED: %s in %s' %
                                            (galPrefix, filter))
                            logging.warning('###    Err: %s - %s' %
                                            (galPrefix, errMsg))
                            with open(logFile, "a") as logMatch:
                                try:
                                    logFormat = "%25s  %20s  %s  FAIL \n"
                                    logMatch.write(logFormat % (galPrefix,
                                                                suffixSmall,
                                                                filter))
                                    fcntl.flock(logMatch, fcntl.LOCK_UN)
                                except IOError:
                                    pass
                            gc.collect()
                    else:
                        logging.warning('### SMALLMASK is FAILED for %s' %
                                        galPrefix)
                        logging.warning('###    Err: %s - Can not find %s' %
                                        (galPrefix, mskSmall))
                        with open(logFile, "a") as logMatch:
                            try:
                                logFormat = "%25s  %20s  %s  NMSK \n"
                                logMatch.write(logFormat % (galPrefix,
                                                            suffixSmall,
                                                            filter))
                                fcntl.flock(logMatch, fcntl.LOCK_UN)
                            except IOError:
                                pass

                    """ Large Mask """
                    if os.path.isfile(mskLarge):
                        if args.verbose:
                            print "##     Input MaskLarge : %s" % mskLarge
                        suffixLarge = ellipSuffix + '_msklarge'
                        try:
                            cSbp.coaddCutoutSbp(galPrefix, root=galRoot,
                                                verbose=args.verbose,
                                                psf=False,
                                                inEllip=inEllipBin,
                                                bkgCor=args.bkgCor,
                                                zp=args.zp,
                                                step=4,
                                                galX0=args.galX0,
                                                galY0=args.galY0,
                                                galQ0=args.galQ0,
                                                galPA0=args.galPA0,
                                                galRe=args.galRe,
                                                checkCenter=args.noCheckCenter,
                                                updateIntens=args.updateIntens,
                                                pix=args.pix,
                                                plot=args.plot,
                                                redshift=args.redshift,
                                                olthresh=args.olthresh,
                                                fracBad=args.fracBad,
                                                lowClip=args.lowClip,
                                                uppClip=args.uppClip,
                                                nClip=args.nClip,
                                                intMode=args.intMode,
                                                minIt=args.minIt,
                                                maxIt=args.maxIt,
                                                maxTry=args.maxTry,
                                                outRatio=args.outRatio,
                                                exMask=mskLarge,
                                                suffix=suffixLarge,
                                                plMask=args.plmask,
                                                imgSub=args.imgSub,
                                                isophote=args.isophote,
                                                xttools=args.xttools)
                            logging.info('### LARGEMASK is DONE for %s in %s' %
                                         (galPrefix, filter))
                            with open(logFile, "a") as logMatch:
                                try:
                                    logFormat = "%25s  %20s  %s  DONE \n"
                                    logMatch.write(logFormat % (galPrefix,
                                                                suffixLarge,
                                                                filter))
                                    fcntl.flock(logMatch, fcntl.LOCK_UN)
                                except IOError:
                                    pass
                            gc.collect()
                        except Exception, errMsg:
                            print WAR
                            print str(errMsg)
                            print WAR
                            logging.warning('### LARGEMASK FAILED: %s in %s' %
                                            (galPrefix, filter))
                            logging.warning('###    Err: %s - %s' %
                                            (galPrefix, errMsg))
                            with open(logFile, "a") as logMatch:
                                try:
                                    logFormat = "%25s  %20s  %s  FAIL \n"
                                    logMatch.write(logFormat % (galPrefix,
                                                                suffixLarge,
                                                                filter))
                                    fcntl.flock(logMatch, fcntl.LOCK_UN)
                                except IOError:
                                    pass
                            gc.collect()
                    else:
                        logging.warning('### LARGEMASK is FAILED for %s' %
                                        galPrefix)
                        logging.warning('###    Err: %s - Can not find %s' %
                                        (galPrefix, mskLarge))
                        with open(logFile, "a") as logMatch:
                            try:
                                logFormat = "%25s  %20s  %s  NMSK \n"
                                logMatch.write(logFormat % (galPrefix,
                                                            suffixLarge,
                                                            filter))
                                fcntl.flock(logMatch, fcntl.LOCK_UN)
                            except IOError:
                                pass
            except Exception, errMsg:
                print WAR
                print str(errMsg)
                print WAR
                warnings.warn('### The 1-D SBP is failed for %s in %s' %
                              (galPrefix, filter))
                logging.warning('### The 1-D SBP is FAILED for %s in %s' %
                                (galPrefix, filter))
                logging.warning('###    Err: %s - %s' % (galPrefix, errMsg))
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s  %20s  %s  FAIL \n"
                        logMatch.write(logFormat % (galPrefix,
                                                    ellipSuffix,
                                                    filter))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass

                gc.collect()
    else:
        raise Exception("### Can not find the input catalog: %s" % args.incat)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Prefix of the galaxy image files")
    parser.add_argument("incat", help="The input catalog for cutout")
    parser.add_argument("filter", help="Filter to analysis")
    parser.add_argument('-r', '--rerun', dest='rerun',
                        help="Name of the rerun", default='default')
    parser.add_argument('-i', '--id', dest='id',
                        help="Name of the column for galaxy ID",
                        default='ID')
    parser.add_argument('-mf', '--mFilter', dest='maskFilter',
                        help="Filter for Mask", default=None)
    parser.add_argument('-rf', '--rFilter', dest='refFilter',
                        help="Reference filter for Ellipse run",
                        default='HSC-I')
    parser.add_argument('-rr', '--rRerun', dest='refRerun',
                        help="Reference rerun for Ellipse run",
                        default=None)
    parser.add_argument('-rm', '--rModel', dest='refModel',
                        help="Reference ellipse binary output",
                        default='3')
    parser.add_argument('--multiMask', dest='multiMask',
                        action="store_true",
                        help='Run Force mode using multiple masks',
                        default=False)
    parser.add_argument("--suffix",
                        help="Suffix of the output file",
                        default='')
    parser.add_argument('--sample', dest='sample', help="Sample name",
                        default=None)
    parser.add_argument('-j', '--njobs', type=int,
                        help='Number of jobs run at the same time',
                        dest='njobs', default=1)
    parser.add_argument('--plmask', dest='plmask', action="store_true",
                        default=True)
    parser.add_argument('--imgSub', dest='imgSub', action="store_true",
                        default=False)
    parser.add_argument("--isophote", dest='isophote',
                        help="Location of the x_isophote.e file",
                        default=None)
    parser.add_argument("--xttools", dest='xttools',
                        help="Location of the x_ttools.e file",
                        default=None)
    """ Optional """
    parser.add_argument("--intMode", dest='intMode',
                        help="Method for integration",
                        default='mean')
    parser.add_argument('--pix', dest='pix', help='Pixel Scale',
                        type=float, default=0.168)
    parser.add_argument('--step', dest='step', help='Step size',
                        type=float, default=0.10)
    parser.add_argument('--zp', dest='zp', help='Photometric zeropoint',
                        type=float, default=27.0)
    parser.add_argument('--redshift', dest='redshift',
                        help='Photometric zeropoint',
                        type=float, default=None)
    parser.add_argument('--olthresh', dest='olthresh',
                        help='Central locator threshold',
                        type=float, default=0.30)
    parser.add_argument('--uppClip', dest='uppClip',
                        help='Upper limit for clipping',
                        type=float, default=3.0)
    parser.add_argument('--lowClip', dest='lowClip',
                        help='Upper limit for clipping',
                        type=float, default=3.0)
    parser.add_argument('--nClip', dest='nClip',
                        help='Upper limit for clipping',
                        type=int, default=2)
    parser.add_argument('--fracBad', dest='fracBad',
                        help='Outer threshold',
                        type=float, default=0.5)
    parser.add_argument('--minIt', dest='minIt',
                        help='Minimum number of iterations',
                        type=int, default=20)
    parser.add_argument('--maxIt', dest='maxIt',
                        help='Maximum number of iterations',
                        type=int, default=150)
    parser.add_argument('--maxTry', dest='maxTry',
                        help='Maximum number of attempts of ellipse run',
                        type=int, default=4)
    parser.add_argument('--galX0', dest='galX0',
                        help='Center X0',
                        type=float, default=None)
    parser.add_argument('--galY0', dest='galY0', help='Center Y0',
                        type=float, default=None)
    parser.add_argument('--galQ0', dest='galQ0', help='Input Axis Ratio',
                        type=float, default=None)
    parser.add_argument('--galPA0', dest='galPA0', help='Input Position Angle',
                        type=float, default=None)
    parser.add_argument('--galRe', dest='galRe',
                        help='Input Effective Radius in pixel',
                        type=float, default=None)
    parser.add_argument('--outRatio', dest='outRatio',
                        help='Increase the boundary of SBP by this ratio',
                        type=float, default=1.2)
    parser.add_argument('--verbose', dest='verbose', action="store_true",
                        default=False)
    parser.add_argument('--psf', dest='psf', action="store_true",
                        help='Ellipse run on PSF', default=True)
    parser.add_argument('--plot', dest='plot', action="store_true",
                        help='Generate summary plot', default=True)
    parser.add_argument('--bkgCor', dest='bkgCor', action="store_true",
                        help='Background correction', default=False)
    parser.add_argument('--noCheckCenter', dest='noCheckCenter',
                        action="store_false",
                        help='Check if the center is off', default=True)
    parser.add_argument('--updateIntens', dest='updateIntens',
                        action="store_true",
                        default=True)

    args = parser.parse_args()

    run(args)
