#!/usr/bin/env python
# encoding: utf-8
"""Run ELLIPSE to extract 1-D SBP in batch mode."""

import os
import gc
import glob
import logging
import warnings
import argparse

try:
    import psutil
    psutilOk = True
except Exception:
    psutilOk = False
from astropy.io import fits

import coaddCutoutSbp as cSbp

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def run(args):
    """
    Run coaddCutoutSbp in batch mode.

    Parameters
    """
    if psutilOk:
        proc = psutil.Process(os.getpid())
        gc.collect()
        mem0 = proc.memory_info().rss
    else:
        gc.collect()

    if os.path.isfile(args.incat):
        data = fits.open(args.incat)[1].data
        id = (args.id)
        rerun = (args.rerun).strip()
        prefix = (args.prefix).strip()
        suffix = (args.suffix).strip()
        filter = (args.filter).strip().upper()

        """ Keep a log """
        logSuffix = '_%s_%s_sbp.log' % (filter, rerun)
        logFile = (args.incat).replace('.fits', logSuffix)
        logging.basicConfig(filename=logFile)

        print COM
        print "## Will deal with %d galaxies ! " % len(data)

        for index, galaxy in enumerate(data):
            print COM
            galID = str(galaxy[id]).strip()
            galPrefix = prefix + '_' + galID + '_' + filter + '_full'
            galRoot = os.path.join(galID, filter)
            print "## Deal with %s now : %i / %i" % (galID,
                                                     (index + 1),
                                                     len(data))
            print COM
            if not os.path.isdir(galRoot):
                logging.warning('### Can not find ' +
                                'ROOT folder for %s' % galRoot)
                continue
            fitsList = glob.glob(os.path.join(galRoot, '*.fits'))
            if len(fitsList) < 2:
                logging.warning('### MISSING Data in %s' % galRoot)
                continue

            galImg = galPrefix + '_img.fits'
            if (not os.path.isfile(os.path.join(galRoot, galImg)) and not
                    os.path.islink(os.path.join(galRoot, galImg))):
                logging.warning('### Can not find ' +
                                'CUTOUT IMAGE for %s in %s' %
                                (galPrefix, filter))
                continue
            """
            Set up a rerun
            """
            galRoot = os.path.join(galRoot, rerun.strip())
            if not os.path.isdir(galRoot):
                os.makedirs(galRoot)
            """ Link the necessary files to the rerun folder """
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
                print "###  Use %s filter for mask \n" % mskFilter
                mskPrefix = prefix + '_' + galID + '_' + mskFilter + '_full'
                mskRoot = os.path.join(galID, mskFilter, rerun)
                galMsk = os.path.join(mskRoot, mskPrefix + '_mskfin.fits')
                if not os.path.isfile(galMsk):
                    logging.warning('### Can not find ' +
                                    'MASK for  %s in %s' %
                                    (galPrefix, filter))
                    continue
            else:
                galMsk = None

            if suffix == '':
                ellipSuffix = rerun
            else:
                ellipSuffix = rerun + '_' + suffix

            print '\n' + SEP
            try:
                cSbp.coaddCutoutSbp(galPrefix, root=galRoot,
                                    verbose=args.verbose,
                                    psf=args.psf,
                                    inEllip=args.inEllip,
                                    bkgCor=args.bkgCor,
                                    zp=args.zp,
                                    step=args.step,
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
                                    multiEllipse=args.multiEllipse,
                                    suffix=ellipSuffix,
                                    plMask=args.plmask,
                                    noMask=args.nomask,
                                    imgSub=args.imgSub)

                logging.info('### The 1-D SBP is DONE for %s in %s' %
                             (galPrefix, filter))
                print SEP
            except Exception, errMsg:
                print str(errMsg)
                warnings.warn('### The 1-D SBP is failed for %s in %s' %
                              (galPrefix, filter))
                logging.warning('### The 1-D SBP is FAILED for %s in %s' %
                                (galPrefix, filter))
                logging.warning('###     Error :%s' % errMsg)
                print SEP + '\n'

            if psutilOk:
                mem1 = proc.memory_info().rss
                gc.collect()
                mem2 = proc.memory_info().rss
            else:
                gc.collect()

    else:
        raise Exception("### Can not find the input catalog: %s" % args.incat)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Prefix of the galaxy image files")
    parser.add_argument("incat", help="The input catalog for cutout")
    parser.add_argument('-i', '--id', dest='id',
                        help="Name of the column for galaxy ID",
                        default='ID')
    parser.add_argument('-f', '--filter', dest='filter', help="Filter",
                        default='HSC-I')
    parser.add_argument('-mf', '--mFilter', dest='maskFilter',
                        help="Filter for Mask", default=None)
    parser.add_argument('-r', '--rerun', dest='rerun',
                        help="Name of the rerun", default='default')
    parser.add_argument("--suffix",
                        help="Suffix of the output file",
                        default='')
    """ Optional """
    parser.add_argument("--intMode", dest='intMode',
                        help="Method for integration",
                        default='median')
    parser.add_argument('--inEllip', dest='inEllip',
                        help='Input Ellipse table',
                        default=None)
    parser.add_argument('--pix', dest='pix', help='Pixel Scale',
                        type=float, default=0.168)
    parser.add_argument('--step', dest='step', help='Step size',
                        type=float, default=0.20)
    parser.add_argument('--zp', dest='zp', help='Photometric zeropoint',
                        type=float, default=27.0)
    parser.add_argument('--redshift', dest='redshift',
                        help='Photometric zeropoint',
                        type=float, default=None)
    parser.add_argument('--olthresh', dest='olthresh',
                        help='Central locator threshold',
                        type=float, default=0.50)
    parser.add_argument('--uppClip', dest='uppClip',
                        help='Upper limit for clipping',
                        type=float, default=2.5)
    parser.add_argument('--lowClip', dest='lowClip',
                        help='Upper limit for clipping',
                        type=float, default=2.5)
    parser.add_argument('--nClip', dest='nClip',
                        help='Upper limit for clipping',
                        type=int, default=3)
    parser.add_argument('--fracBad', dest='fracBad',
                        help='Fraction of bad pixels allowed',
                        type=float, default=0.7)
    parser.add_argument('--minIt', dest='minIt',
                        help='Minimum number of iterations',
                        type=int, default=30)
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
                        default=True)
    parser.add_argument('--psf', dest='psf', action="store_true",
                        help='Ellipse run on PSF', default=True)
    parser.add_argument('--plot', dest='plot', action="store_true",
                        help='Generate summary plot', default=True)
    parser.add_argument('--bkgCor', dest='bkgCor', action="store_true",
                        help='Background correction', default=True)
    parser.add_argument('--noCheckCenter', dest='noCheckCenter',
                        action="store_false",
                        help='Check if the center is off', default=True)
    parser.add_argument('--updateIntens', dest='updateIntens',
                        action="store_true",
                        default=True)
    parser.add_argument('--multiEllipse', dest='multiEllipse',
                        action="store_true",
                        default=False)
    parser.add_argument('--plmask', dest='plmask', action="store_true",
                        default=True)
    parser.add_argument('--nomask', dest='nomask', action="store_true",
                        default=False)
    parser.add_argument('--imgSub', dest='imgSub', action="store_true",
                        default=False)

    args = parser.parse_args()

    run(args)
