#!/usr/bin/env python
# encoding: utf-8
"""Prepare the HSC cutout for photometry."""

import os
import fcntl
import logging
import argparse
import warnings

from astropy.io import fits

import coaddCutoutPrepare as ccp

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def run(args):
    """
    Run coaddCutoutPrepare in batch mode.

    Parameters:
    """
    if os.path.isfile(args.incat):
        """ Basic information """
        data = fits.open(args.incat)[1].data
        id = (args.id)
        rerun = (args.rerun).strip()
        prefix = (args.prefix).strip()
        filter = (args.filter).strip().upper()
        """ Keep a log """
        if args.sample is not None:
            logPre = prefix + '_' + args.sample
        else:
            logPre = prefix
        logFile = logPre + '_prep_' + filter + '.log'
        if not os.path.isfile(logFile):
            os.system('touch ' + logFile)

        if args.verbose:
            print "## Will deal with %d galaxies ! " % len(data)

        for galaxy in data:
            """ Galaxy ID and prefix """
            galID = str(galaxy[id]).strip()
            galPrefix = prefix + '_' + galID + '_' + filter + '_full'
            if args.verbose:
                print "## Will Deal with %s now ! " % galID
            """Folder for the data"""
            galRoot = os.path.join(galID, filter)
            if not os.path.isdir(galRoot):
                warnings.warn('### Cannot find folder %s' % galRoot)
                with open(logFile, "a") as logMatch:
                    logStr = "%25s  %10s  NDIR \n"
                    try:
                        logMatch.write(logStr % (galPrefix, rerun))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
                continue
            """Image"""
            galImg = galPrefix + '_img.fits'
            if not os.path.isfile(os.path.join(galRoot, galImg)):
                warnings.warn('### Cannot find image %s' % galImg)
                with open(logFile, "a") as logMatch:
                    logStr = "%25s  %10s  NIMG \n"
                    try:
                        logMatch.write(logStr % (galPrefix, rerun))
                        fcntl.flock(logMatch, fcntl.LOCK_UN)
                    except IOError:
                        pass
                continue

            try:
                if rerun == 'default':
                    ccp.coaddCutoutPrepare(
                        galPrefix,
                        root=galRoot,
                        rerun='default',
                        bSizeH=10.0,
                        bSizeC=60.0,
                        thrH=2.5,
                        thrC=1.5,
                        galR1=1.6,
                        galR2=3.1,
                        galR3=6.5,
                        growH=2.1,
                        growW=4.0,
                        growC=5.0,
                        sigma=6.0,
                        sigthr=0.02,
                        minDetH=8,
                        minDetC=16,
                        debThrH=32,
                        debThrC=48,
                        debConH=0.0005,
                        debConC=0.0005,
                        kernel=4,
                        central=1,
                        maskMethod=1,
                        growMethod=1,
                        useSigArr=False,
                        noBkgC=False,
                        noBkgH=False,
                        combBad=True,
                        combDet=True,
                        multiMask=args.multiMask)
                    with open(logFile, "a") as logMatch:
                        logStr = "%25s  %10s  DONE \n"
                        try:
                            logMatch.write(logStr % (galPrefix, rerun))
                            fcntl.flock(logMatch, fcntl.LOCK_UN)
                        except IOError:
                            pass
                elif rerun == 'smallR1':
                    ccp.coaddCutoutPrepare(
                        galPrefix,
                        root=galRoot,
                        rerun='smallR1',
                        bSizeH=10.0,
                        bSizeC=40.0,
                        thrH=2.5,
                        thrC=1.1,
                        growH=2.5,
                        growW=5.5,
                        growC=7.5,
                        galR1=1.4,
                        galR2=2.5,
                        galR3=4.0,
                        sigma=9.0,
                        sigthr=0.01,
                        kernel=4,
                        central=1,
                        maskMethod=1,
                        growMethod=1,
                        useSigArr=False,
                        noBkgC=False,
                        noBkgH=False,
                        minDetH=5,
                        minDetC=8,
                        debThrH=16,
                        debThrC=32,
                        debConH=0.001,
                        debConC=0.0025,
                        combBad=True,
                        combDet=True,
                        multiMask=False)
                    with open(logFile, "a") as logMatch:
                        logStr = "%25s  %10s  DONE \n"
                        try:
                            logMatch.write(logStr % (galPrefix, rerun))
                            fcntl.flock(logMatch, fcntl.LOCK_UN)
                        except IOError:
                            pass
                elif rerun == 'largeR1':
                    ccp.coaddCutoutPrepare(
                        galPrefix,
                        root=galRoot,
                        rerun='largeR1',
                        bSizeH=10.0,
                        bSizeC=40.0,
                        thrH=3.0,
                        thrC=1.5,
                        growH=1.5,
                        growW=3.0,
                        growC=4.5,
                        galR1=2.5,
                        galR2=5.0,
                        galR3=7.0,
                        sigma=7.0,
                        sigthr=0.02,
                        kernel=4,
                        central=1,
                        maskMethod=1,
                        growMethod=1,
                        useSigArr=False,
                        noBkgC=False,
                        noBkgH=False,
                        minDetH=5,
                        minDetC=8,
                        debThrH=16,
                        debThrC=32,
                        debConH=0.001,
                        debConC=0.0025,
                        combBad=True,
                        combDet=True,
                        multiMask=False)
                    with open(logFile, "a") as logMatch:
                        logStr = "%25s  %10s  DONE \n"
                        try:
                            logMatch.write(logStr % (galPrefix, rerun))
                            fcntl.flock(logMatch, fcntl.LOCK_UN)
                        except IOError:
                            pass
                else:
                    ccp.coaddCutoutPrepare(
                        galPrefix,
                        root=galRoot,
                        rerun=rerun,
                        bSizeH=args.bSizeH,
                        bSizeC=args.bSizeC,
                        thrH=args.thrH,
                        thrC=args.thrC,
                        growH=args.growH,
                        growW=args.growW,
                        growC=args.growC,
                        kernel=args.kernel,
                        central=args.central,
                        maskMethod=args.mask,
                        growMethod=args.grow,
                        useSigArr=args.useSigArr,
                        noBkgC=args.noBkgC,
                        noBkgH=args.noBkgH,
                        minDetH=args.minDetH,
                        minDetC=args.minDetC,
                        debThrH=args.debThrH,
                        debThrC=args.debThrC,
                        debConH=args.debConH,
                        debConC=args.debConC,
                        combBad=args.combBad,
                        combDet=args.combDet,
                        multiMask=args.multiMask)
                    with open(logFile, "a") as logMatch:
                        logStr = "%25s  %10s  DONE \n"
                        try:
                            logMatch.write(logStr % (galPrefix, rerun))
                            fcntl.flock(logMatch, fcntl.LOCK_UN)
                        except IOError:
                            pass
            except Exception, errMsg:
                print WAR
                print str(errMsg)
                print WAR
                warnings.warn('### The preparation is failed for %s in %s' %
                              (galPrefix, filter))
                logging.warning('### The preparation is failed for %s in %s' %
                                (galPrefix, filter))
                with open(logFile, "a") as logMatch:
                    logStr = "%25s  %10s  FAIL \n"
                    try:
                        logMatch.write(logStr % (galPrefix, rerun))
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
        '-r',
        '--rerun',
        dest='rerun',
        help="Name of the rerun",
        default='default')
    parser.add_argument(
        '--multiMask', dest='multiMask', action="store_true", default=False)
    parser.add_argument(
        '--sample', dest='sample', help="Sample name", default=None)
    parser.add_argument(
        '--verbose',
        dest='verbose',
        help="Verbose",
        default=True,
        action="store_true")
    """ Optional """
    parser.add_argument(
        '-k',
        dest='kernel',
        help='SExtractor detection kernel',
        type=int,
        default=4,
        choices=range(1, 8))
    parser.add_argument(
        '-c',
        dest='central',
        help='Method to clean the central region',
        type=int,
        default=1,
        choices=range(1, 4))
    parser.add_argument(
        '-m',
        dest='mask',
        help='Method to grow the All object mask',
        type=int,
        default=1,
        choices=range(1, 4))
    parser.add_argument(
        '-g',
        dest='grow',
        help='Method to grow the Final object mask',
        type=int,
        default=1,
        choices=range(1, 3))
    parser.add_argument(
        '--bkgH',
        dest='bSizeH',
        help='Background size for the Hot Run',
        type=int,
        default=8)
    parser.add_argument(
        '--bkgC',
        dest='bSizeC',
        help='Background size for the Cold Run',
        type=int,
        default=40)
    parser.add_argument(
        '--thrH',
        dest='thrH',
        help='Detection threshold for the Hot Run',
        type=float,
        default=2.2)
    parser.add_argument(
        '--thrC',
        dest='thrC',
        help='Detection threshold for the Cold Run',
        type=float,
        default=1.2)
    parser.add_argument(
        '--growC',
        dest='growC',
        help='Ratio of Growth for the Cold Objects',
        type=float,
        default=6.0)
    parser.add_argument(
        '--growW',
        dest='growW',
        help='Ratio of Growth for the Warm Objects',
        type=float,
        default=4.0)
    parser.add_argument(
        '--growH',
        dest='growH',
        help='Ratio of Growth for the Hot Objects',
        type=float,
        default=2.0)
    parser.add_argument(
        '--minDetC',
        dest='minDetC',
        help='Minimum pixels for Cold Detections',
        type=float,
        default=8.0)
    parser.add_argument(
        '--minDetH',
        dest='minDetH',
        help='Minimum pixels for Hot Detections',
        type=float,
        default=5.0)
    parser.add_argument(
        '--debThrC',
        dest='debThrC',
        help='Deblending threshold for the Cold Run',
        type=float,
        default=32.0)
    parser.add_argument(
        '--debThrH',
        dest='debThrH',
        help='Deblending threshold for the Hot Run',
        type=float,
        default=16.0)
    parser.add_argument(
        '--debConC',
        dest='debConC',
        help='Deblending continuum level for the Cold Run',
        type=float,
        default=0.001)
    parser.add_argument(
        '--debConH',
        dest='debConH',
        help='Deblending continuum level for the Hot Run',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--galR1',
        dest='galR1',
        help='galR1 = galR1 * galR90',
        type=float,
        default=2.0)
    parser.add_argument(
        '--galR2',
        dest='galR2',
        help='galR2 = galR2 * galR90',
        type=float,
        default=4.0)
    parser.add_argument(
        '--galR3',
        dest='galR3',
        help='galR3 = galR3 * galR90',
        type=float,
        default=6.0)
    parser.add_argument(
        '--sigma',
        dest='sigma',
        help='Sigma to Gaussian smooth the segmentation image',
        type=float,
        default=6.0)
    parser.add_argument(
        '--noBkgC', dest='noBkgC', action="store_true", default=False)
    parser.add_argument(
        '--noBkgH', dest='noBkgH', action="store_true", default=False)
    parser.add_argument(
        '--useSigArr', dest='useSigArr', action="store_true", default=False)
    parser.add_argument(
        '--combBad', dest='combBad', action="store_true", default=False)
    parser.add_argument(
        '--combDet', dest='combDet', action="store_true", default=False)
    args = parser.parse_args()

    run(args)
