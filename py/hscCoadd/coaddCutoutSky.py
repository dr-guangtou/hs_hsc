#!/usr/bin/env python
# encoding: utf-8
"""Estimate background of HSC cutout."""

from __future__ import (division, print_function)

import os
import copy
import warnings
import argparse

import numpy as np
import scipy
from scipy.stats import sigmaclip

# Astropy
from astropy.io import fits
try:
    from astropy.visualization import hist
    astroHist = True
except ImportError:
    astroHist = False

# SEP
import sep

# Personal
import hscUtils as hUtil
import coaddCutoutPrepare as cdPrep

# Matplotlib related
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = 12, 10
mpl.rcParams['xtick.major.size'] = 8.0
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.size'] = 4.0
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 8.0
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.size'] = 4.0
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rc('axes', linewidth=2)
import matplotlib.pyplot as plt
plt.ioff()

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def readCutout(prefix, root=None, exMask=None, verbose=False):
    """
    Read Cutout Image.

    Parameters:
    """
    # Get the names of necessary input images
    imgFile = prefix + '_img.fits'
    if root is not None:
        imgFile = os.path.join(root, imgFile)
    if exMask is None:
        mskFile = prefix + '_mskall.fits'
        if root is not None:
            mskFile = os.path.join(root, mskFile)
    else:
        mskFile = exMask
    if verbose:
        print("###    Mask Used : %s" % mskFile)

    if os.path.islink(imgFile):
        imgOri = os.readlink(imgFile)
        imgFile = imgOri
    if os.path.islink(mskFile):
        mskOri = os.readlink(mskFile)
        mskFile = mskOri

    if (not os.path.isfile(imgFile)) or (not os.path.isfile(mskFile)):
        print(imgFile, mskFile)
        raise Exception("### Can not find the Image or BadMask File!")
    else:
        imgHdu = fits.open(imgFile)
        imgArr = imgHdu[0].data
        # Header
        imgHead = imgHdu[0].header
        # All objects mask
        mskHdu = fits.open(mskFile)
        mskArr = mskHdu[0].data
    return imgArr, imgHead, mskArr


def showSkyHist(skypix,
                skypix2=None,
                skypix3=None,
                sbExpt=None,
                pngName='skyhist.png',
                skyAvg=None,
                skyStd=None,
                skyMed=None,
                skySkw=None,
                savePng=True):
    """
    Plot the distribution of sky pixels.

    Parameters:
    """
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(hspace=0.07, wspace=0.0,
                        left=0.06, bottom=0.15, top=0.99, right=0.95)
    fontsize = 12
    ax.minorticks_on()

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    if astroHist:
        _ = hist(skypix, bins='knuth', ax=ax, alpha=0.4, color='cyan',
                 histtype='stepfilled', normed=True)
    else:
        _ = plt.hist(skypix, bins=100, ax=ax, alpha=0.4, color='cyan',
                     histtype='stepfilled', normed=True)

    if skypix2 is not None:
        if astroHist:
            _ = hist(skypix2, bins='knuth', ax=ax, alpha=0.9, color='k',
                     histtype='step', normed=True, linewidth=2)
        else:
            _ = plt.hist(skypix2, bins=100, ax=ax, alpha=0.9, color='k',
                         histtype='step', normed=True, linewidth=2)

    if skypix3 is not None:
        if astroHist:
            _ = hist(skypix3, bins='knuth', ax=ax, alpha=0.8, color='k',
                     histtype='step', normed=True, linewidth=2,
                     linestyle='dashed')
        else:
            _ = plt.hist(skypix3, bins=100, ax=ax, alpha=0.8, color='k',
                         histtype='step', normed=True, linewidth=2,
                         linestyle='dashed')
    # Horizontal line
    ax.axvline(0.0, linestyle='-', color='k', linewidth=1.5)

    # Basic properties of the sky pixels
    skyMin = np.nanmin(skypix)
    skyMax = np.nanmax(skypix)
    if skyAvg is None:
        skyAvg = np.nanmean(skypix)
    if skyStd is None:
        skyStd = np.nanstd(skypix)
    if skyMed is None:
        skyMed = np.nanmedian(skypix)
        if not np.isfinite(skyMed):
            skyMed = np.median(skypix)
    if skySkw is None:
        skySkw = scipy.stats.skew(skypix)
    # Highligh the mode of sky pixel distribution
    ax.axvline(skyMed, linestyle='--', color='b', linewidth=1.5)

    ax.set_xlabel('Pixel Value', fontsize=12)
    ax.set_xlim(skyAvg - 4.0 * skyStd, skyAvg + 5.0 * skyStd)
    # Show a few information
    ax.text(
        0.7, 0.9, "Min : %8.4f" % skyMin, fontsize=12, transform=ax.transAxes)
    ax.text(
        0.7, 0.8, "Max : %8.4f" % skyMax, fontsize=12, transform=ax.transAxes)
    ax.text(
        0.7, 0.7, "Avg : %8.4f" % skyAvg, fontsize=12, transform=ax.transAxes)
    ax.text(
        0.7, 0.6, "Std : %8.4f" % skyStd, fontsize=12, transform=ax.transAxes)
    ax.text(
        0.7, 0.5, "Med : %8.4f" % skyMed, fontsize=12, transform=ax.transAxes)
    ax.text(
        0.7, 0.4, "Skew: %8.4f" % skySkw, fontsize=12, transform=ax.transAxes)
    if sbExpt is not None:
        ax.text(0.7, 0.3, "S.B : %8.5f" % sbExpt, fontsize=12,
                transform=ax.transAxes)

    if savePng:
        fig.savefig(pngName, dpi=70)
        plt.close(fig)


def getSEPSky(imgArr,
              mskArr,
              imgHead,
              skyClip=3,
              zp=27.0,
              pix=0.168,
              rebin=4,
              prefix='sep_sky',
              suffix='imgsub',
              verbose=True,
              visual=True,
              bkgSize=40,
              bkgFilter=5,
              saveBkg=False,
              nClip=2):
    """
    Estimating the background using SEP.

    Parameters:
    """
    dimX, dimY = imgArr.shape
    mskX, mskY = mskArr.shape
    if (dimX != mskX) or (dimY != mskY):
        raise Exception("## The image and mask don't have the same size!")

    # What if there is no useful masked pixel
    try:
        sepBkg = sep.Background(imgArr, mask=mskArr, bw=bkgSize, bh=bkgSize,
                                fw=bkgFilter, fh=bkgFilter)
    except ValueError:
        imgArr = imgArr.byteswap(True).newbyteorder()
        sepBkg = sep.Background(imgArr, mask=mskArr, bw=bkgSize, bh=bkgSize,
                                fw=bkgFilter, fh=bkgFilter)

    avgBkg = sepBkg.globalback
    rmsBkg = sepBkg.globalrms
    if (not np.isfinite(avgBkg)) or (not np.isfinite(rmsBkg)):
        warnings.warn("###    The SEP background has problem")
    if verbose:
        print("###    SEP BKG AVG, RMS : %10.7f, %10.7f" % (avgBkg, rmsBkg))

    # Subtract the sky model from the image
    try:
        imgBkg = sepBkg.back()
        imgSub = (imgArr - imgBkg)
        fitsSub = prefix + '_' + suffix + '.fits'
        # Save the new image
        imgSave = copy.deepcopy(imgSub)
        imgSave = imgSave.byteswap(True).newbyteorder()
        hdu = fits.PrimaryHDU(imgSave)
        hdu.header = imgHead
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(fitsSub, overwrite=True)

        if saveBkg:
            fitsBkg = prefix + '_' + suffix + '_bkg.fits'
            bkgSave = copy.deepcopy(imgBkg)
            bkgSave = bkgSave.byteswap(True).newbyteorder()
            hdu = fits.PrimaryHDU(bkgSave)
            hdu.header = imgHead
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(fitsBkg, overwrite=True)
    except Exception:
        warnings.warn("## Something wrong with the SEP background subtraction")

    # Rebin image
    dimBinX = int((dimX - 1) / rebin)
    dimBinY = int((dimY - 1) / rebin)
    try:
        imgBin = hUtil.congrid(imgArr, (dimBinX, dimBinY), method='nearest')
        subBin = hUtil.congrid(imgSub, (dimBinX, dimBinY), method='nearest')
        mskBin = hUtil.congrid(mskArr, (dimBinX, dimBinY), method='neighbour')
    except Exception:
        warnings.warn("congrid fails!")
        print("###    Image rebin is failed for this galaxy !!!")
        imgBin = imgArr
        subBin = imgSub
        mskBin = mskArr

    pixSky1 = imgBin[mskBin == 0].flatten()
    pixSky1 = pixSky1[np.isfinite(pixSky1)]
    try:
        pixSky1, low1, upp1 = sigmaclip(pixSky1, low=skyClip, high=skyClip)
    except Exception:
        warnings.warn("\nSigma clip fails for imgBin")

    pixSky2 = subBin[mskBin == 0].flatten()
    pixSky2 = pixSky2[np.isfinite(pixSky2)]
    try:
        pixSky2, low2, upp2 = sigmaclip(pixSky2, low=skyClip, high=skyClip)
    except Exception:
        warnings.warn("Sigma clip fails for mskBin")

    if visual:
        sepPNG = prefix + '_' + suffix + '_skyhist.png'
        showSkyHist(pixSky2, skypix2=pixSky1, pngName=sepPNG)

    return imgSub


def getGlobalSky(imgArr,
                 mskAll,
                 skyClip=3,
                 zp=27.0,
                 pix=0.168,
                 rebin=4,
                 prefix='coadd_sky',
                 suffix='global_',
                 verbose=True,
                 visual=True,
                 nClip=2):
    """
    Estimate the Global Sky.

    Estimating the global sky background level by using the mean
    of a rebined image

    This could also be used to estimate the expect surface brightness
    limit of the image

    """
    # Estimate the global background level
    dimX, dimY = imgArr.shape

    # Pixel values of all pixels that are not masked out (before rebinned)
    pixels = imgArr[mskAll == 0].flatten()
    pixels = pixels[np.isfinite(pixels)]
    try:
        pixNoMsk, low3, upp3 = sigmaclip(pixels, low=skyClip, high=skyClip)
    except Exception:
        warnings.warn("\n### sigmaclip failed for original image!")
        pixNoMsk = pixels
        del pixels

    try:
        # Rebin image
        dimBinX = int((dimX - 1) / rebin)
        dimBinY = int((dimY - 1) / rebin)
        imgBin = hUtil.congrid(imgArr, (dimBinX, dimBinY), method='nearest')
        mskBin = hUtil.congrid(mskAll, (dimBinX, dimBinY), method='neighbour')
    except Exception:
        warnings.warn('### congrid failed!')
        print("\n###    Image rebin is failed for this galaxy !!!")
        imgBin = imgArr
        mskBin = mskAll

    # Get all the pixels that are not masked out
    pixels = imgBin[mskBin == 0].flatten()
    pixels = pixels[np.isfinite(pixels)]
    try:
        pixNoMskBin, low4, upp4 = sigmaclip(pixels, low=skyClip, high=skyClip)
    except Exception:
        warnings.warn("### sigmaclip failed for binned image!")
        pixNoMskBin = pixels

    numSkyPix = len(pixNoMskBin)
    # Get the basic statistics of the global sky
    skyAvg, skyStd = np.nanmean(pixNoMskBin), np.nanstd(pixNoMskBin)
    if not np.isfinite(skyAvg) or not np.isfinite(skyStd):
        warnings.warn("\n###    No useful global skyAvg / Std for %s" % prefix)
    skyMed = np.nanmedian(pixNoMskBin)
    if not np.isfinite(skyMed):
        warnings.warn("\n###    No useful global skyMed for %s" % prefix)
        skyMed = skyAvg if np.isfinite(skyAvg) else 0.00

    skySkw = scipy.stats.skew(pixNoMskBin)
    sbExpt = cdPrep.getSbpValue(3.0 * skyStd, pix * rebin, pix * rebin, zp=zp)

    if not np.isfinite(sbExpt):
        warnings.warn("\n###    No useful global sbExpt for %s" % prefix)
    if verbose:
        print("###    Median / Mean / Std / Skew / SBP: " +
              " %8.5f / %8.5f / %8.5f / %8.5f / %5.2f" % (skyMed, skyAvg,
                                                          skyStd, skySkw,
                                                          sbExpt))

    if visual:
        skyPNG = prefix + '_' + suffix + 'skyhist.png'
        showSkyHist(
            pixNoMskBin,
            skypix2=pixNoMsk,
            sbExpt=sbExpt,
            pngName=skyPNG,
            skyAvg=skyAvg,
            skyMed=skyMed,
            skyStd=skyStd,
            skySkw=skySkw)

    """Save a txt file summary"""
    skyTxt = prefix + '_' + suffix + 'sky.dat'
    text_file = open(skyTxt, "w")
    text_file.write("IMAGE: %s \n" % prefix)
    text_file.write("REBIN: %3d \n" % rebin)
    text_file.write("NSKYPIX: %10d \n" % numSkyPix)
    text_file.write("SKYMED: %10.6f \n" % skyMed)
    text_file.write("SKYAVG: %10.6f \n" % skyAvg)
    text_file.write("SKYSTD: %10.6f \n" % skyStd)
    text_file.write("SKYSKW: %10.6f \n" % skySkw)
    text_file.write("SBEXPT: %10.6f \n" % sbExpt)
    text_file.close()

    return numSkyPix, skyMed, skyAvg, skyStd, skySkw, sbExpt


def coaddCutoutSky(prefix,
                   root=None,
                   verbose=True,
                   skyClip=3.0,
                   pix=0.168,
                   zp=27.0,
                   rebin=6,
                   visual=True,
                   exMask=None,
                   bkgSize=40,
                   bkgFilter=5,
                   saveBkg=False,
                   nClip=2):
    """
    Estimate the Sky Background for Coadd Image.

    Parameters:
    """
    # 0. Get necessary information
    # Read the input cutout image
    imgArr, imgHead, mskArr = readCutout(prefix, root=root, exMask=exMask)
    if (root is not None) and (root[-1] != '/'):
        root += '/'
    if verbose:
        print(SEP)
        print("###    DEAL WITH IMAGE : %s" % (root + prefix + '_img.fits'))
        print("###    Binning: %3d; %4.1f sigma-clipping" % (rebin, skyClip))

    # Get rid of the NaN pixels, if there is any
    mskArr[np.isnan(imgArr)] = 1

    # 1. SEP Sky
    imgSub = getSEPSky(
        imgArr,
        mskArr,
        imgHead,
        skyClip=skyClip,
        zp=zp,
        pix=pix,
        rebin=rebin,
        prefix=(root + prefix),
        suffix='imgsub',
        verbose=True,
        visual=True,
        bkgSize=bkgSize,
        bkgFilter=bkgFilter,
        saveBkg=saveBkg,
        nClip=nClip)

    # 2. Global Background Estimation
    suffixGlob = 'rebin' + str(rebin).strip() + '_'
    skyGlobal = getGlobalSky(
        imgSub,
        mskArr,
        skyClip=skyClip,
        zp=zp,
        pix=pix,
        rebin=rebin,
        prefix=(root + prefix),
        suffix=suffixGlob,
        visual=visual,
        verbose=verbose,
        nClip=nClip)

    return skyGlobal


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Prefix of the cutout image files")
    parser.add_argument(
        '-r',
        '--root',
        dest='root',
        help='Path to the image files',
        default=None)
    parser.add_argument(
        '-m', '--mask', help="External file for image mask", default=None)
    parser.add_argument(
        '--skyclip',
        dest='skyClip',
        help='Sigma for pixel clipping',
        type=float,
        default=3.0)
    parser.add_argument(
        '--rebin',
        dest='rebin',
        help='Rebin the image by N x N pixels',
        type=int,
        default=6)
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

    coaddCutoutSky(
        args.prefix,
        root=args.root,
        pix=args.pix,
        zp=args.zp,
        rebin=args.rebin,
        skyClip=args.skyClip,
        verbose=args.verbose,
        visual=args.visual,
        exMask=args.mask,
        bkgSize=args.bkgSize,
        bkgFilter=args.bkgFilter,
        saveBkg=args.saveBkg,
        nClip=args.nClip)
