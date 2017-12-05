#!/usr/bin/env python
# encoding: utf-8
"""Estimate background for HSC cutouts."""

from __future__ import (division, print_function)

import os
import glob
import fcntl
import logging
import argparse
import warnings

import numpy as np

from astropy.io import fits

import coaddCutoutSky as ccs

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def run(args):
    """
    Run coaddCutoutSky in batch mode.

    Parameters:
    """
    if os.path.isfile(args.incat):
        """ Basic information """
        id = (args.id)
        data = fits.open(args.incat)[1].data
        rerun = (args.rerun).strip()
        prefix = (args.prefix).strip()
        filter = (args.filter).strip().upper()
        """ Keep a log """
        if args.sample is not None:
            logPre = prefix + '_' + args.sample
        else:
            logPre = prefix
        logFile = logPre + '_sky_' + filter + '.log'
        if not os.path.isfile(logFile):
            os.system('touch ' + logFile)
        if args.verbose:
            print("\n## Will deal with %d galaxies ! " % len(data))

        for galaxy in data:
            """ Galaxy ID and prefix """
            galID = str(galaxy[id]).strip()
            galPrefix = prefix + '_' + galID + '_' + filter + '_full'
            """Folder for the data"""
            galRoot = os.path.join(galID, filter)
            if not os.path.isdir(galRoot):
                if args.verbose:
                    print('\n### Cannot find the folder: %s !' % galRoot)
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s  %5s NDIR  %6.1f  %7.4f  %7.4f" + \
                                    "  %7.4f  %7.4f  %7.4f  \n"
                        logMatch.write(logFormat %
                                       (galPrefix, filter, np.nan, np.nan,
                                        np.nan, np.nan, np.nan, np.nan))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
                continue
            """Collect the FITS file information"""
            fitsList = glob.glob(os.path.join(galRoot, '*.fits'))
            if len(fitsList) <= 3:
                if args.verbose:
                    print("### Missing data under %s" % galRoot)
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s  %5s MISS  %6.1f  %7.4f  %7.4f" + \
                                    "  %7.4f  %7.4f  %7.4f  \n"
                        logMatch.write(logFormat %
                                       (galPrefix, filter, np.nan, np.nan,
                                        np.nan, np.nan, np.nan, np.nan))
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
                    print("\n###  Use %s filter for mask \n" % mskFilter)
                mskPrefix = (prefix + '_' + galID + '_' + mskFilter + '_full')
                mskRoot = os.path.join(galID, mskFilter, rerun)
                galMsk = os.path.join(mskRoot, mskPrefix + '_mskall.fits')
                if not os.path.isfile(galMsk):
                    if args.verbose:
                        print(
                            '\n### Can not find final mask : %s !' % galMsk)
                    with open(logFile, "a") as logMatch:
                        try:
                            logFormat = "%25s  %5s NMSK  %6.1f  %7.4f  " + \
                                        "%7.4f  %7.4f  %7.4f  %7.4f  \n"
                            logMatch.write(logFormat %
                                           (galPrefix, filter, np.nan, np.nan,
                                            np.nan, np.nan, np.nan, np.nan))
                            fcntl.flock(logMatch, fcntl.LOCK_UN)
                        except IOError:
                            pass
                    continue
            else:
                galMsk = None
            """ Start to estimate the sky background """
            try:
                skyGlobal = ccs.coaddCutoutSky(
                    galPrefix,
                    root=galRoot,
                    pix=args.pix,
                    zp=args.zp,
                    rebin=args.rebin,
                    skyClip=args.skyClip,
                    verbose=args.verbose,
                    visual=args.visual,
                    exMask=galMsk,
                    bkgSize=args.bkgSize,
                    bkgFilter=args.bkgFilter,
                    saveBkg=args.saveBkg,
                    nClip=args.nClip)
                numSkyPix, skyMed, skyAvg, skyStd, skySkw, sbExpt = skyGlobal
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s  %5s  %3d  %6d  %7.4f  %7.4f" + \
                                    "  %7.4f  %7.4f  %7.4f  \n"
                        logMatch.write(logFormat %
                                       (galPrefix, filter, args.rebin,
                                        numSkyPix, skyMed, skyAvg, skyStd,
                                        skySkw, sbExpt))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
            except Exception, errMsg:
                print(WAR)
                print(str(errMsg))
                warnings.warn('### The sky estimate is failed ' +
                              'for %s in %s' % (galID, filter))
                logging.warning('### The sky estimate is failed ' +
                                'for %s in %s' % (galID, filter))
                with open(logFile, "a") as logMatch:
                    try:
                        logFormat = "%25s  %5s  %3d  %6d  %7.4f  %7.4f" + \
                                    "  %7.4f  %7.4f  %7.4f  \n"
                        logMatch.write(logFormat %
                                       (galPrefix, filter, args.rebin, 0,
                                        np.nan, np.nan, np.nan, np.nan,
                                        np.nan))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
    else:
        raise Exception("### Can not find the input catalog: %s" % args.incat)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Prefix of the galaxy image files")
    parser.add_argument("incat", help="The input catalog for cutout")
    parser.add_argument(
        '-i',
        '--id',
        dest='id',
        help="Name of the column for galaxy ID",
        default='index')
    parser.add_argument(
        '-f', '--filter', dest='filter', help="Filter", default='HSC-I')
    parser.add_argument(
        '-mf',
        '--mFilter',
        dest='maskFilter',
        help="Filter for Mask",
        default=None)
    parser.add_argument(
        '-r',
        '--rerun',
        dest='rerun',
        help="Name of the rerun",
        default='default')
    parser.add_argument(
        '--sample', dest='sample', help="Sample name", default=None)
    """ Optional """
    parser.add_argument(
        '--skyclip',
        dest='skyClip',
        help='Sigma for pixel clipping',
        type=float,
        default=3.0)
    parser.add_argument(
        '--bkgSize',
        dest='bkgSize',
        help='Background size for SEP',
        type=int,
        default=60)
    parser.add_argument(
        '--bkgFilter',
        dest='bkgFilter',
        help='Background filter size for SEP',
        type=int,
        default=5)
    parser.add_argument(
        '--rebin',
        dest='rebin',
        help='Rebin the image by N x N pixels',
        type=int,
        default=6)
    parser.add_argument(
        '--pix',
        dest='pix',
        help='Pixel scale of the iamge',
        type=float,
        default=0.168)
    parser.add_argument(
        '--zp',
        dest='zp',
        help='Photometric zeropoint of the image',
        type=float,
        default=27.0)
    parser.add_argument(
        '--verbose', dest='verbose', action="store_true", default=True)
    parser.add_argument(
        '--visual', dest='visual', action="store_true", default=True)
    parser.add_argument(
        '--nClip',
        dest='nClip',
        help='Number of iterations for clipping',
        type=int,
        default=2)
    parser.add_argument(
        '--saveBkg', dest='saveBkg', action="store_true", default=False)

    args = parser.parse_args()

    run(args)
