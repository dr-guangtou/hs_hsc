#!/usr/bin/env python
# encoding: utf-8
"""Extract 1-D SBP for HSC cutout."""

from __future__ import (division, print_function)

import os
import gc
import copy
import warnings
import argparse

import numpy as np
# Astropy
from astropy.io import fits

# Personal
import hscUtils as hUtil
import galSBP

# Matplotlib related
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = 12, 10
mpl.rcParams['xtick.major.size'] = 10.0
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.minor.size'] = 5.0
mpl.rcParams['xtick.minor.width'] = 2.5
mpl.rcParams['ytick.major.size'] = 10.0
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['ytick.minor.size'] = 5.0
mpl.rcParams['ytick.minor.width'] = 2.5
mpl.rc('axes', linewidth=3.0)
import matplotlib.pyplot as plt

# For high-contrast image
try:
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('k', 1.)
except Exception:
    from palettable.cubehelix import perceptual_rainbow_16
    cmap = perceptual_rainbow_16.mpl_colormap
    cmap.set_bad('k', 1.)

SEP = '-' * 100


def readSbpInput(prefix, root=None, exMask=None, imgSub=True):
    """
    Read in necessary files for Ellipse run.

    Parameters:
    """
    # Get the names of necessary input images
    if imgSub:
        imgFile = prefix + '_imgsub.fits'
    else:
        imgFile = prefix + '_img.fits'
    """ Root DIR """
    if root is not None:
        imgFile = os.path.join(root, imgFile)
    """ External mask"""
    if exMask is None:
        if root is not None:
            mskFile = prefix + '_mskfin.fits'
            mskFile = os.path.join(root, mskFile)
    else:
        mskFile = exMask
    """ Input Image """
    if imgSub and (not os.path.isfile(imgFile)):
        # Fall back to original image
        warnings.warn("# Can not find the background subtracted image!")
        imgFile = os.path.join(root, (prefix + '_img.fits'))

    if os.path.islink(imgFile):
        imgOri = os.readlink(imgFile)
    else:
        imgOri = imgFile

    """ Mask Image """
    if os.path.islink(mskFile):
        mskOri = os.readlink(mskFile)
    else:
        mskOri = mskFile
    if not os.path.isfile(imgOri):
        raise Exception("### Cannot find the input image : %s !" % imgOri)
    if not os.path.isfile(mskOri):
        raise Exception("### Can not find the input mask : %s !" % mskOri)
    # Image
    imgHdu = fits.open(imgOri)
    imgArr = imgHdu[0].data
    imgHead = imgHdu[0].header
    # Mask
    mskHdu = fits.open(mskOri)
    mskArr = mskHdu[0].data
    mskHead = mskHdu[0].header

    return imgFile, imgArr, imgHead, mskFile, mskArr, mskHead


def readPsfInput(prefix, root=None):
    """
    Read in the PSF file.

    Parameters:
    """
    psfFile = prefix + '_psf.fits'

    if root is not None:
        psfFile = os.path.join(root, psfFile)

    if os.path.islink(psfFile):
        psfOri = os.readlink(psfFile)

    if not os.path.isfile(psfOri):
        raise Exception("### Can not find the input psf image : %s !" % psfOri)

    # PSF
    psfHdu = fits.open(psfOri)
    psfArr = psfHdu[0].data
    psfDimX, psfDimY = psfArr.shape

    return psfFile, psfDimX, psfDimY


def readInputSky(prefix, root=None, rebin='rebin6'):
    """
    Read in the Input Sky Result.

    Parameters:
    """
    skyFile = prefix + '_' + rebin + '_sky.dat'
    if root is not None:
        skyFile = os.path.join(root, skyFile)

    if not os.path.isfile(skyFile):
        raise Exception("### Can not find the input \
                sky summary : %s !" % skyFile)

    skySum = open(skyFile, 'r').readlines()
    skyMed = float(skySum[3].split(':')[1].strip())
    skyAvg = float(skySum[4].split(':')[1].strip())
    skyStd = float(skySum[5].split(':')[1].strip())

    return skyMed, skyAvg, skyStd


def imgSameSize(img1, img2):
    """Check whether images have the same size."""
    dimX1, dimY1 = img1.shape
    dimX2, dimY2 = img2.shape

    return (dimX1 == dimX2) and (dimY1 == dimY2)


def ellipCompare(ellipStack, outPng='ellipse_compare.png',
                 zp=27.0, maxRad=None, pix=0.168, bkg=0.0,
                 exptime=1.0, outRatio=1.2, pngSize=10,
                 ellipLabel=None):
    """
    Compare the Ellipse results.

    Parameters:
    """
    reg1 = [0.1, 0.1, 0.88, 0.88]
    fig = plt.figure(figsize=(pngSize, pngSize))
    ax1 = fig.add_axes(reg1)

    """ ax1 SBP """
    ax1.minorticks_on()
    ax1.invert_yaxis()
    ax1.tick_params(axis='both', which='major', labelsize=20, pad=8)

    radStr = 'RSMA (arcsec$^{1/4}$)'
    ax1.set_xlabel(radStr, fontsize=23)
    ax1.set_ylabel('${\mu}$ (mag/arcsec$^2$)', fontsize=26)

    ellipLine = ['-', '--', '--', '-.', '-.', '-.']
    ellipColor = ['k', 'r', 'b', 'g', 'c', 'm']
    for ii, ellip in enumerate(ellipStack):
        if ellipLabel is None:
            label = str(ii+1)
        else:
            label = ellipLabel[ii]
        if ellip is not None:
            ax1.plot(ellip['rsma'], ellip['sbp_cor'],
                     linestyle=ellipLine[ii],
                     c=ellipColor[ii], linewidth=3.5, alpha=0.9,
                     label=label)
    ax1.set_xlim(0.49, 4.4)
    ax1.set_ylim(30.5, 16.9)

    ax1.legend(loc=[0.60, 0.65], fontsize=20)

    """ Save Figure """
    fig.savefig(outPng, dpi=80)
    plt.close(fig)

    return


def ellipSummary(ellipOut1, ellipOut2, ellipOut3, image,
                 maxRad=None, mask=None, radMode='rsma',
                 outPng='ellipse_summary.png', zp=27.0, threshold=None,
                 psfOut=None, useKpc=None, pix=0.168, verbose=False,
                 showZoom=True, exptime=1.0, bkg=0.0, outRatio=1.2,
                 pngSize=16, imgType='_imgsub', dpi=120):
    """
    Make a summary plot for the ellipse run.

    Parameters:
    """
    """ Left side: SBP """
    reg1 = [0.08, 0.07, 0.45, 0.33]
    reg2 = [0.08, 0.40, 0.45, 0.15]
    reg3 = [0.08, 0.55, 0.45, 0.15]
    reg4 = [0.08, 0.70, 0.45, 0.15]
    reg5 = [0.08, 0.85, 0.45, 0.14]
    """ Right side: Curve of growth & IsoMap """
    reg6 = [0.60, 0.07, 0.38, 0.29]
    reg7 = [0.60, 0.36, 0.38, 0.15]
    reg8 = [0.60, 0.55, 0.38, 0.39]

    fig = plt.figure(figsize=(pngSize, pngSize))
    """ Left """
    ax1 = fig.add_axes(reg1)
    ax2 = fig.add_axes(reg2)
    ax3 = fig.add_axes(reg3)
    ax4 = fig.add_axes(reg4)
    ax5 = fig.add_axes(reg5)
    """ Right """
    ax6 = fig.add_axes(reg6)
    ax7 = fig.add_axes(reg7)
    ax8 = fig.add_axes(reg8)

    """ Image """
    img = fits.open(image)[0].data
    imgX, imgY = img.shape
    imgMsk = copy.deepcopy(img)

    if mask is not None:
        msk = fits.open(mask)[0].data
        imgMsk[msk > 0] = np.nan

    """ Find the proper outer boundary """
    sma = ellipOut3['sma']
    radOuter = galSBP.ellipseGetOuterBoundary(ellipOut3, ratio=outRatio,
                                              threshold=threshold,
                                              polyOrder=12)
    if not np.isfinite(radOuter):
        if verbose:
            print("###     radOuter is NaN, use 0.80 * max(SMA) !")
        radOuter = np.nanmax(sma) * 0.80
    else:
        if verbose:
            print("###  OutRadius", radOuter)
    indexUse1 = np.where(ellipOut1['sma'] <= (radOuter*1.2))
    indexUse2 = np.where(ellipOut2['sma'] <= (radOuter*1.2))
    indexUse3 = np.where(ellipOut3['sma'] <= (radOuter*1.2))
    curveOri = ellipOut3['growth_ori']
    curveSub = ellipOut3['growth_sub']
    curveCor = ellipOut3['growth_cor']
    growthCurveOri = -2.5 * np.log10(curveOri) + zp
    growthCurveCor = -2.5 * np.log10(curveCor) + zp

    maxIsoFluxOri = np.nanmax(curveOri[indexUse3])
    magFluxOri100 = -2.5 * np.log10(maxIsoFluxOri) + zp
    if verbose:
        print("###     MagTot ORI : ", magFluxOri100)
    ax1.text(0.55, 0.85,
             '$\mathrm{mag}_{\mathrm{tot,ori}}=%5.2f$' % magFluxOri100,
             fontsize=30, transform=ax1.transAxes)

    maxIsoFluxSub = np.nanmax(curveSub[indexUse3])
    magFluxSub100 = -2.5 * np.log10(maxIsoFluxSub) + zp
    if verbose:
        print("###     MagTot SUB : ", magFluxSub100)
    """
    ax1.text(0.55, 0.78,
             '$\mathrm{mag}_{\mathrm{tot,sub}}=%5.2f$' % magFluxSub100,
             fontsize=24, transform=ax1.transAxes)
    """

    maxIsoFluxCor = np.nanmax(curveCor[indexUse3])
    magFlux50 = -2.5 * np.log10(maxIsoFluxCor * 0.50) + zp
    magFlux100 = -2.5 * np.log10(maxIsoFluxCor) + zp
    if verbose:
        print("###     MagTot COR : ", magFlux100)
    ax1.text(0.55, 0.72,
             '$\mathrm{mag}_{\mathrm{tot,cor}}=%5.2f$' % magFlux100,
             fontsize=30, transform=ax1.transAxes)

    """ Type of Radius """
    if radMode is 'rsma':
        if useKpc is None:
            radStr = '$R^{1/4}\ (\mathrm{arcsec}^{1/4})$'
            rad1 = ellipOut1['rsma_asec']
            rad2 = ellipOut2['rsma_asec']
            rad3 = ellipOut3['rsma_asec']
        else:
            radStr = '$R^{1/4}\ (\mathrm{kpc}^{1/4})$'
            rad1 = (ellipOut1['sma_asec']*useKpc)**0.25
            rad2 = (ellipOut2['sma_asec']*useKpc)**0.25
            rad3 = (ellipOut3['sma_asec']*useKpc)**0.25

        minRad = 0.41 if 0.41 >= np.nanmin(rad3) else np.nanmin(rad3)
        if useKpc is None:
            radOut = (radOuter * pix * 1.2) ** 0.25
        else:
            radOut = (radOuter * pix * useKpc * 1.2) ** 0.25
        if maxRad is None:
            maxSma = np.nanmax(ellipOut3['rsma'])
            maxRad = np.nanmax(rad3)
        else:
            maxSma = maxRad
            if useKpc is None:
                maxRad = (maxRad * pix) ** 0.25
            else:
                maxRad = (maxRad * pix * useKpc) ** 0.25
    elif radMode is 'sma':
        if useKpc is None:
            radStr = '$R\ (\mathrm{arcsec})$'
            rad1 = ellipOut1['sma_asec']
            rad2 = ellipOut2['sma_asec']
            rad3 = ellipOut3['sma_asec']
        else:
            radStr = '$R\ (\mathrm{kpc})$'
            rad1 = ellipOut1['sma_asec'] * useKpc
            rad2 = ellipOut2['sma_asec'] * useKpc
            rad3 = ellipOut3['sma_asec'] * useKpc

        minRad = 0.05 if 0.05 >= np.nanmin(rad3) else np.nanmin(rad3)
        if useKpc is None:
            radOut = radOuter * pix * 1.2
        else:
            radOut = radOuter * pix * useKpc * 1.2
        if maxRad is None:
            maxSma = np.nanmax(ellipOut3['sma'])
            maxRad = np.nanmax(rad3)
        else:
            maxSma = maxRad
            if useKpc is None:
                maxRad = maxRad * pix
            else:
                maxRad = maxRad * pix * useKpc
    elif radMode is 'log':
        sma1 = ellipOut1['sma_asec']
        sma2 = ellipOut2['sma_asec']
        sma3 = ellipOut3['sma_asec']
        if useKpc is None:
            radStr = '$\log\ (R/\mathrm{arcsec})$'
            rad1 = np.log10(sma1)
            rad2 = np.log10(sma2)
            rad3 = np.log10(sma3)
        else:
            radStr = '$\log\ (R/\mathrm{kpc})$'
            rad1 = np.log10(sma1 * useKpc)
            rad2 = np.log10(sma2 * useKpc)
            rad3 = np.log10(sma3 * useKpc)

        minRad = -1.2 if -1.2 >= np.nanmin(rad3) else np.nanmin(rad3)
        if useKpc is None:
            radOut = np.log10(radOuter * pix * 1.2)
        else:
            radOut = np.log10(radOuter * pix * useKpc * 1.2)
        if maxRad is None:
            maxSma = np.nanmax(ellipOut3['sma'])
            maxRad = np.nanmax(rad3)
        else:
            maxSma = maxRad
            if useKpc is None:
                maxRad = np.log10(maxRad * pix)
            else:
                maxRad = np.log10(maxRad * pix * useKpc)
    else:
        raise Exception('### Wrong type of Radius: sma, rsma, log')

    """ ax1 SBP """
    ax1.minorticks_on()
    ax1.invert_yaxis()
    ax1.tick_params(axis='both', which='major', labelsize=22, pad=8)

    ax1.set_xlabel(radStr, fontsize=30)
    ax1.set_ylabel('${\mu}\ (\mathrm{mag}/\mathrm{arcsec}^2)$',
                   fontsize=28)

    sbp_sub = ellipOut3['sbp_sub']
    sbp_ori = ellipOut3['sbp_ori']
    sbp_cor = ellipOut3['sbp_cor']
    sbp_low = ellipOut3['sbp_low']
    sbp_upp = ellipOut3['sbp_upp']

    maxIsoSbp = np.nanmax(sbp_sub)
    if verbose:
        print("###    MaxIsoSbp : ", maxIsoSbp)
    ax1.fill_between(rad3[indexUse3],
                     (sbp_cor[indexUse3] + sbp_upp[indexUse3] -
                      sbp_sub[indexUse3]),
                     (sbp_cor[indexUse3] - sbp_ori[indexUse3] +
                      sbp_low[indexUse3]),
                     facecolor='r', alpha=0.2)
    ax1.plot(rad3[indexUse3], sbp_ori[indexUse3], '--', color='k',
             linewidth=3.5)
    """
    ax1.plot(rad3[indexUse3], sbp_sub[indexUse3], '-.', color='b',
             linewidth=3.0)
    """
    ax1.plot(rad3[indexUse3], sbp_cor[indexUse3], '-', color='r',
             linewidth=4.0)
    sbpBuffer1 = 0.75
    sbpBuffer2 = 0.50
    minSbp = np.nanmin(ellipOut3['sbp_low'][indexUse3]) - sbpBuffer1
    maxSbp = np.nanmax(ellipOut3['sbp_upp'][indexUse3]) + sbpBuffer2
    maxSbp = maxSbp if maxSbp >= 29.0 else 28.9
    maxSbp = maxSbp if maxSbp <= 30.9 else 30.89
    if psfOut is not None:
        if radMode is 'rsma':
            psfRad = psfOut['rsma_asec']
        elif radMode is 'sma':
            psfRad = psfOut['sma_asec']
        elif radMode is 'log':
            psfRad = psfOut['sma_asec']
            psfRad = np.log10(psfOut['sma_asec'])
        else:
            raise Exception('### Wrong type of Radius: sma, rsma, log')
        psfSbp = psfOut['sbp'] - (np.nanmin(psfOut['sbp']) -
                                  np.nanmin(ellipOut3['sbp']))
        ax1.plot(psfRad, psfSbp, '-', color='k', linewidth=3.0, alpha=0.7)
    ax1.set_xlim(minRad, radOut)
    ax1.set_ylim(maxSbp, minSbp)

    """ ax2 Ellipticity """
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax2.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax2.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax2.locator_params(axis='y', tight=True, nbins=4)
    ax2.set_ylabel('$e$', fontsize=34)
    ax2.axhline((1.0 - ellipOut2['avg_q'][0]), color='k', linestyle='--',
                linewidth=3.0)
    ax2.fill_between(rad2[indexUse2],
                     (ellipOut2['ell'][indexUse2] +
                      ellipOut2['ell_err'][indexUse2]),
                     (ellipOut2['ell'][indexUse2] -
                      ellipOut2['ell_err'][indexUse2]),
                     facecolor='r', alpha=0.2)
    ax2.plot(rad2[indexUse2], ellipOut2['ell'][indexUse2], '-', color='r',
             linewidth=3.0)
    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2.set_xlim(minRad, radOut)
    ellBuffer = 0.06
    minEll = np.nanmin(ellipOut2['ell'][indexUse2]) - ellBuffer
    maxEll = np.nanmax(ellipOut2['ell'][indexUse2]) + ellBuffer
    ax2.set_ylim(minEll, maxEll)

    """ ax3 PA """
    ax3.minorticks_on()
    ax3.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax3.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax3.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax3.locator_params(axis='y', tight=True, nbins=4)
    ax3.set_ylabel('$\mathrm{PA}\ (\mathrm{deg})$',  fontsize=23)
    medPA = np.nanmedian(ellipOut2['pa_norm'][indexUse2])
    avgPA = ellipOut2['avg_pa'][0]
    if (avgPA-medPA >= 85.0) and (avgPA <= 92.0):
        avgPA -= 180.0
    elif (avgPA-medPA <= -85.0) and (avgPA >= -92.0):
        avgPA += 180.0
    ax3.axhline(avgPA, color='k', linestyle='--', linewidth=3.0)
    ax3.fill_between(rad2[indexUse2],
                     (ellipOut2['pa_norm'][indexUse2] +
                      ellipOut2['pa_err'][indexUse2]),
                     (ellipOut2['pa_norm'][indexUse2] -
                      ellipOut2['pa_err'][indexUse2]),
                     facecolor='r', alpha=0.2)
    ax3.plot(rad2[indexUse2], ellipOut2['pa_norm'][indexUse2], '-', color='r',
             linewidth=3.0)
    ax3.xaxis.set_major_formatter(NullFormatter())
    ax3.set_xlim(minRad, radOut)
    paBuffer = 10.0
    minPA = np.nanmin(ellipOut2['pa_norm'][indexUse2])-paBuffer
    maxPA = np.nanmax(ellipOut2['pa_norm'][indexUse2])+paBuffer
    minPA = minPA if minPA >= -110.0 else -100.0
    maxPA = maxPA if maxPA <= 110.0 else 100.0
    ax3.set_ylim(minPA, maxPA)

    """ ax4 X0/Y0 """
    ax4.minorticks_on()
    ax4.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax4.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax4.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax4.locator_params(axis='y', tight=True, nbins=4)
    ax4.set_ylabel('$\mathrm{X}_{0}\ \mathrm{or}\ $' +
                   '$\mathrm{Y}_{0}\ (\mathrm{pix})$', fontsize=23)
    ax4.axhline(ellipOut1['avg_x0'][0], linestyle='--', color='r', alpha=0.6,
                linewidth=3.0)
    ax4.fill_between(rad1[indexUse1],
                     (ellipOut1['x0'][indexUse1] +
                      ellipOut1['x0_err'][indexUse1]),
                     (ellipOut1['x0'][indexUse1] -
                      ellipOut1['x0_err'][indexUse1]),
                     facecolor='r', alpha=0.20)
    ax4.plot(rad1[indexUse1], ellipOut1['x0'][indexUse1], '-', color='r',
             linewidth=3.0, label='X0')
    ax4.axhline(ellipOut1['avg_y0'][0], linestyle='-.', color='b', alpha=0.6,
                linewidth=3.0)
    ax4.fill_between(rad1[indexUse1],
                     (ellipOut1['y0'][indexUse1] +
                     ellipOut1['y0_err'][indexUse1]),
                     (ellipOut1['y0'][indexUse1] -
                     ellipOut1['y0_err'][indexUse1]),
                     facecolor='b', alpha=0.25)
    ax4.plot(rad1[indexUse1], ellipOut1['y0'][indexUse1], '-', color='b',
             linewidth=3.0, label='Y0')
    ax4.xaxis.set_major_formatter(NullFormatter())
    ax4.set_xlim(minRad, radOut)
    xBuffer = 3.0
    minX0 = np.nanmin(ellipOut1['x0'][indexUse1])
    maxX0 = np.nanmax(ellipOut1['x0'][indexUse1])
    minY0 = np.nanmin(ellipOut1['y0'][indexUse1])
    maxY0 = np.nanmax(ellipOut1['y0'][indexUse1])
    minCen = minX0 if minX0 <= minY0 else minY0
    maxCen = maxX0 if maxX0 >= maxY0 else maxY0
    ax4.set_ylim(minCen-xBuffer, maxCen+xBuffer)

    """ ax5 A4/B4 """
    ax5.minorticks_on()
    ax5.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax5.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax5.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax5.locator_params(axis='y', tight=True, nbins=4)
    ax5.set_ylabel('$a_4\ \mathrm{or}\ b_4$',  fontsize=22)
    ax5.axhline(0.0, linestyle='-', color='k', alpha=0.4)
    ax5.fill_between(rad2[indexUse2],
                     (ellipOut2['a4'][indexUse2] +
                     ellipOut2['a4_err'][indexUse2]),
                     (ellipOut2['a4'][indexUse2] -
                     ellipOut2['a4_err'][indexUse2]),
                     facecolor='r', alpha=0.20)
    ax5.plot(rad2[indexUse2], ellipOut2['a4'][indexUse2], '-',
             color='r', linewidth=3.0, label='A4')

    ax5.fill_between(rad2[indexUse2],
                     (ellipOut2['b4'][indexUse2] +
                     ellipOut2['b4_err'][indexUse2]),
                     (ellipOut2['b4'][indexUse2] -
                     ellipOut2['b4_err'][indexUse2]),
                     facecolor='b', alpha=0.20)
    ax5.plot(rad2[indexUse2], ellipOut2['b4'][indexUse2], '-',
             color='b', linewidth=3.0, label='B4')

    ax5.xaxis.set_major_formatter(NullFormatter())
    ax5.set_xlim(minRad, radOut)

    abBuffer = 0.02
    minA4 = np.nanmin(ellipOut2['a4'][indexUse2])
    minB4 = np.nanmin(ellipOut2['b4'][indexUse2])
    maxA4 = np.nanmax(ellipOut2['a4'][indexUse2])
    maxB4 = np.nanmax(ellipOut2['b4'][indexUse2])
    minAB = minA4 if minA4 <= minB4 else minB4
    maxAB = maxA4 if maxA4 >= maxB4 else maxB4
    ax5.set_ylim(minAB-abBuffer, maxAB+abBuffer)

    """ ax6 Growth Curve """
    ax6.minorticks_on()
    ax6.tick_params(axis='both', which='major', labelsize=16, pad=8)
    ax6.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax6.yaxis.set_major_locator(MaxNLocator(prune='upper'))

    ax6.set_xlabel(radStr, fontsize=30)
    ax6.set_ylabel('$\mathrm{Curve\ of\ Growth}\ (\mathrm{mag})$',
                   fontsize=20)
    ax6.axhline(magFlux100, linestyle='-', color='k',
                alpha=0.6, linewidth=3.0, label='$\mathrm{mag}_{100}$')
    ax6.axhline(magFlux50,  linestyle='--', color='k',
                alpha=0.6, linewidth=3.0, label='$\mathrm{mag}_{50}$')
    ax6.axvline(radOut, linestyle='-', color='g', alpha=0.6, linewidth=4.0)

    ax6.plot(rad3, growthCurveOri, '--', color='k', linewidth=3.5,
             label='$\mathrm{CoG}_{\mathrm{ori}}$')
    """
    ax6.plot(rad3, growthCurveSub, '-.', color='b', linewidth=3.5,
             label='$\mathrm{CoG}_{\mathrm{sub}$')
    ax6.axvline(imgR50, linestyle='-', color='g', alpha=0.4, linewidth=3.0)
    """
    ax6.plot(rad3, growthCurveCor, '-', color='r', linewidth=4.0,
             label='$\mathrm{CoG}_{\mathrm{cor}}$')
    ax6.legend(loc=[0.65, 0.07], shadow=True, fancybox=True,
               fontsize=20)
    minCurve = (magFlux100 - 0.9)
    maxCurve = (magFlux100 + 2.9)
    curveUse = growthCurveOri[np.isfinite(growthCurveOri)]
    radTemp = rad3[np.isfinite(growthCurveOri)]
    radInner = radTemp[curveUse <= maxCurve][0]
    """
    maxCurve = (np.nanmax(growthCurveOri[np.isfinite(growthCurveOri)])
                - 1.2)
    """
    ax6.set_xlim((radInner - 0.02), (maxRad + 0.2))
    ax6.set_ylim(maxCurve, minCurve)

    """ ax7 Intensity Curve """
    ax7.minorticks_on()
    ax7.tick_params(axis='both', which='major', labelsize=16, pad=10)
    ax7.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax7.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax7.locator_params(axis='y', tight=True, nbins=4)
    ax7.axhline(0.0, linestyle='-', color='k', linewidth=2.5, alpha=0.8)
    ax7.fill_between(rad3,
                     (rad3 * 0.0 - 1.0 * np.nanmedian(ellipOut3['int_err'])),
                     (rad3 * 0.0 + 1.0 * np.nanmedian(ellipOut3['int_err'])),
                     facecolor='k', edgecolor='none', alpha=0.15)

    ax7.fill_between(rad3,
                     (ellipOut3['intens_cor'] - ellipOut3['int_err']),
                     (ellipOut3['intens_cor'] + ellipOut3['int_err']),
                     facecolor='r', alpha=0.3)
    ax7.plot(rad3, ellipOut3['intens'], '--', color='k', linewidth=2.5)
    """
    ax7.plot(rad3, ellipOut3['intens_sub'], '-', color='r', linewidth=3.5)
    ax7.axvline(imgR50, linestyle='-', color='g', alpha=0.4, linewidth=3.0)
    ax7.axhline(bkg, linestyle='--', color='c', alpha=0.6, linewidth=2.5)
    """
    ax7.plot(rad3, ellipOut3['intens_cor'], '-', color='r', linewidth=3.0)
    ax7.axvline(radOut, linestyle='-', color='g', alpha=0.6, linewidth=4.0)

    indexOut = np.where(ellipOut3['intens'] <= (0.003 *
                        np.nanmax(ellipOut3['intens'])))
    minOut = np.nanmin(ellipOut3['intens'][indexOut] -
                       ellipOut3['int_err'][indexOut])
    maxOut = np.nanmax(ellipOut3['intens'][indexOut] +
                       ellipOut3['int_err'][indexOut])
    sepOut = ((maxOut - minOut) / 4.0)
    ax7.xaxis.set_major_formatter(NullFormatter())
    ax7.set_xlim((radInner - 0.02), (maxRad + 0.2))
    ax7.set_ylim(minOut - sepOut, maxOut)

    """ ax8 IsoPlot """
    imgFile = os.path.basename(image)
    imgTitle = imgFile.replace('.fits', '')
    imgTitle = imgTitle.replace(imgType, '')

    ax8.tick_params(axis='both', which='major', labelsize=20)
    ax8.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax8.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax8.set_title(imgTitle, fontsize=25, fontweight='bold')
    ax8.title.set_position((0.5, 1.05))

    galX0 = ellipOut3['avg_x0'][0]
    galY0 = ellipOut3['avg_y0'][0]
    imgSizeX, imgSizeY = img.shape

    if (galX0 > maxSma) and (galY0 > maxSma) and showZoom:
        zoomReg = imgMsk[np.int(galX0-maxSma):np.int(galX0+maxSma),
                         np.int(galY0-maxSma):np.int(galY0+maxSma)]
        # Define the new center of the cropped images
        xPad = (imgSizeX / 2.0 - maxSma)
        yPad = (imgSizeY / 2.0 - maxSma)
    else:
        zoomReg = imgMsk
        xPad = 0
        yPad = 0

    try:
        imin, imax = hUtil.zscale(zoomReg, contrast=0.20, samples=500)
    except Exception:
        imin = np.percentile(np.ravel(showZoom), 0.01)
        imax = np.percentile(np.ravel(showZoom), 0.95)
    # Show the image
    ax8.imshow(np.arcsinh(zoomReg), interpolation="none",
               vmin=imin, vmax=imax, cmap=cmap, origin='lower')
    # Get the Shapes
    ellipIso = galSBP.convIso2Ell(ellipOut3, xpad=xPad, ypad=yPad)
    # Overlay the ellipses on the image
    for ii, e in enumerate(ellipIso):
        if len(ellipIso) >= 30:
            if (ii >= 6) and (ii <= 30) and (ii % 5 == 0):
                ax8.add_artist(e)
                e.set_clip_box(ax8.bbox)
                e.set_alpha(0.4)
                e.set_edgecolor('r')
                e.set_facecolor('none')
                e.set_linewidth(1.0)
            elif (ii > 30):
                ax8.add_artist(e)
                e.set_clip_box(ax8.bbox)
                e.set_alpha(0.9)
                e.set_edgecolor('r')
                e.set_facecolor('none')
                e.set_linewidth(2.0)
        else:
            if (ii >= 6):
                ax8.add_artist(e)
                e.set_clip_box(ax8.bbox)
                e.set_alpha(0.8)
                e.set_edgecolor('r')
                e.set_facecolor('none')
                e.set_linewidth(2.0)

    ellOuter = Ellipse(xy=(ellipOut3['avg_x0'][0],
                       ellipOut3['avg_y0'][0]),
                       height=(radOuter * 2.4),
                       width=(radOuter * 2.4 * ellipOut3['avg_q'][0]),
                       angle=(ellipOut3['avg_pa'][0]))
    ax8.add_artist(ellOuter)
    ellOuter.set_clip_box(ax8.bbox)
    ellOuter.set_alpha(0.9)
    ellOuter.set_edgecolor('g')
    ellOuter.set_facecolor('none')
    ellOuter.set_linewidth(6.0)

    """ Save Figure """
    fig.savefig(outPng, dpi=dpi)
    plt.close(fig)

    return


def coaddCutoutSbp(prefix, root=None, verbose=True, psf=True, inEllip=None,
                   zp=27.0, step=0.12, pix=0.168, exptime=1.0, bkgCor=False,
                   plot=True, galX0=None, galY0=None, galQ0=None, galPA0=None,
                   maxTry=4, galRe=None, redshift=None, psfRecenter=True,
                   showZoom=True, checkCenter=True, updateIntens=True,
                   olthresh=0.5, intMode='mean', lowClip=3.0, uppClip=3.0,
                   nClip=2, fracBad=0.5, minIt=20, maxIt=150, outRatio=1.2,
                   exMask=None, suffix='', plMask=False, noMask=False,
                   multiEllipse=False, imgSub=False,
                   isophote=None, xttools=None):
    """
    Generate 1-D SBP Plot.

    Parameters:
    """
    if verbose:
        print(SEP)
        print("###    CoaddCutoutSbp Start: %s - %s" % (root, prefix))
    gc.collect()

    """ Check the x_isophote.e and x_ttools.e """
    if isophote is not None:
        if not os.path.isfile(isophote):
            raise Exception("# Can not find x_isophote.e: %s" % isophote)
        """ Have to use x_isophote.e and x_ttools.e at the same time """
        if xttools is None:
            raise Exception("# Have to use x_ttools.e at the same time")
        else:
            if not os.path.isfile(xttools):
                raise Exception("# Can not find x_ttools.e: %s" % xttools)
    else:
        print("###    Will use PyRAF for everything !")

    """ 0. Organize Input Data """
    # Read in the input image, mask, psf, and their headers
    sbpInput = readSbpInput(prefix, root=root, exMask=exMask, imgSub=imgSub)
    imgFile, imgArr, imgHead, mskFile, mskArr, mskHead = sbpInput
    if (root is not None) and (root[-1] != '/'):
        root += '/'
    if not imgSameSize(imgArr, mskArr):
        raise Exception("### The Image and Mask need to have EXACTLY \
                same dimensions!")
    if imgSub:
        imgType = '_imgsub'
    else:
        imgType = '_img'
    """
    Delete image and mask array
    """
    del imgArr
    del mskArr

    if os.path.islink(imgFile):
        imgOri = os.readlink(imgFile)
    else:
        imgOri = imgFile

    if os.path.islink(mskFile):
        mskOri = os.readlink(mskFile)
    else:
        mskOri = mskFile

    if noMask:
        mskFile = None
        mskOri = None

    """ 0a. Redshift """
    if redshift is not None:
        scale = hUtil.cosmoScale(redshift)
        useKpc = scale
    else:
        useKpc = None

    """ 0b. Background """
    if bkgCor:
        try:
            skyMed, skyAvg, skyStd = readInputSky(prefix, root=root)
            bkg = skyMed
        except Exception:
            bkg = 0.00
    else:
        bkg = 0.00

    """ 1. Prepare the Input for SBP """
    if (galX0 is None) or (galY0 is None):
        galX, galY = mskHead['GAL_CENX'], mskHead['GAL_CENY']
    else:
        galX, galY = galX0, galY0
    if (galQ0 is None) or (galPA0 is None):
        galQ, galPA = mskHead['GAL_Q'], mskHead['GAL_PA']
    else:
        galQ, galPA = galQ0, galPA0

    galQ = galQ if galQ <= 0.95 else 0.95
    galPA = hUtil.normAngle(galPA, lower=-90.0, upper=90.0, b=True)

    if galRe is None:
        galR50 = mskHead['GAL_R50']
    else:
        galR50 = galRe
    if verbose:
        print("###    Mask  : ", mskFile)
        print("###    X0, Y0, Q0, PA0 : ", galX, galY, galQ, galPA)
        print("###    intMode : ", intMode)

    maxR = mskHead['NAXIS1'] if (mskHead['NAXIS1'] >=
                                 mskHead['NAXIS2']) else mskHead['NAXIS2']
    maxR = (maxR / 2.0) * np.sqrt(2.0)
    if verbose:
        print("###      maxR : ", maxR)

    """
    Actually start to run
    """
    if checkCenter and mskHead['MSK_R20'] == 1:
        raise Exception("### The central region is masked out : %s" % imgFile)
    else:
        """ Ellipse run for the PSF """
        if psf:
            if verbose:
                print("\n###    Ellipse Run on PSF ")
            psfFile = prefix + '_psf.fits'
            if root is not None:
                psfFile = os.path.join(root, psfFile)
            if os.path.islink(psfFile):
                psfOri = os.readlink(psfFile)
            else:
                psfOri = psfFile

            if not os.path.isfile(psfOri):
                raise Exception("### Can not find the \
                                PSF image: %s !" % psfFile)
            psfRes = galSBP.galSBP(psfFile, iniSma=5.0,
                                   pix=pix,
                                   galQ=0.95,
                                   galPA=0.0,
                                   stage=3,
                                   zpPhoto=zp,
                                   recenter=psfRecenter,
                                   outerThreshold=1e-6,
                                   useZscale=False,
                                   savePng=False,
                                   bkg=0.0,
                                   updateIntens=False,
                                   imgType=imgType,
                                   isophote=isophote,
                                   xttools=xttools)
            psfOut, psfBin = psfRes
        else:
            psfOut = None

        """ Ellipse run for the galaxy """
        if inEllip is None:
            iniSma = (galR50 * 2.0)
            """#        Start with Stage 1 """
            if verbose:
                print("\n###    Ellipse Run on Image %s- Stage 1 " % imgFile)
            if isophote is None:
                galSBP.unlearnEllipse()
            ellRes1 = galSBP.galSBP(imgFile,
                                    mask=mskFile,
                                    galX=galX,
                                    galY=galY,
                                    maxSma=maxR,
                                    iniSma=iniSma,
                                    maxTry=maxTry,
                                    galR=galR50,
                                    ellipStep=step,
                                    pix=pix,
                                    bkg=bkg,
                                    galQ=galQ,
                                    galPA=galPA,
                                    stage=1,
                                    zpPhoto=zp,
                                    updateIntens=updateIntens,
                                    olthresh=olthresh,
                                    lowClip=lowClip,
                                    uppClip=uppClip,
                                    nClip=nClip,
                                    fracBad=fracBad,
                                    intMode=intMode,
                                    minIt=minIt,
                                    maxIt=maxIt,
                                    plMask=plMask,
                                    suffix=suffix,
                                    imgType=imgType,
                                    isophote=isophote,
                                    xttools=xttools)
            ellOut1, ellBin1 = ellRes1

            if ellOut1 is None:
                raise Exception("!!!!! ELLIPSE RUN FAILED AT STAGE 1 !!!!")
            if (galX0 is None) or (galY0 is None):
                galX0 = ellOut1['avg_x0'][0]
                galY0 = ellOut1['avg_y0'][0]

            if checkCenter:
                if (np.abs(galX0 -
                           mskHead['NAXIS1']/2.0) >= 20.0) or \
                   (np.abs(galX0 -
                           mskHead['NAXIS1']/2.0) >= 20.0):
                    raise Exception("The Center is Off !")

            """ # Start with Stage 2 """
            if verbose:
                print("\n###    Ellipse Run on Image %s- Stage 2 " % imgFile)
            ellRes2 = galSBP.galSBP(imgFile, mask=mskFile,
                                    galX=galX0,
                                    galY=galY0,
                                    maxSma=maxR,
                                    iniSma=iniSma,
                                    galR=galR50,
                                    ellipStep=step,
                                    maxTry=maxTry,
                                    pix=pix,
                                    bkg=bkg,
                                    galQ=galQ,
                                    galPA=galPA,
                                    stage=2,
                                    zpPhoto=zp,
                                    updateIntens=updateIntens,
                                    olthresh=olthresh,
                                    lowClip=lowClip,
                                    uppClip=uppClip,
                                    nClip=nClip,
                                    fracBad=fracBad,
                                    intMode=intMode,
                                    minIt=minIt,
                                    maxIt=maxIt,
                                    plMask=plMask,
                                    suffix=suffix,
                                    imgType=imgType,
                                    isophote=isophote,
                                    xttools=xttools)
            ellOut2, ellBin2 = ellRes2

            if ellOut2 is None:
                raise Exception("!!!!! ELLIPSE RUN FAILED AT STAGE 2 !!!!")

            if (galQ0 is None) or (galPA0 is None):
                if (ellOut2['avg_q'][0] <= 0.95):
                    galQ0 = ellOut2['avg_q'][0]
                else:
                    galQ0 = galQ

                if np.isfinite(ellOut2['avg_pa'][0]):
                    galPA0 = hUtil.normAngle(ellOut2['avg_pa'][0],
                                             lower=-90.0,
                                             upper=90.0,
                                             b=True)
                else:
                    galPA0 = galPA

            """ # Start with Stage 3 """
            if verbose:
                print("\n###    Ellipse Run on Image %s- Stage 3 " % imgFile)
            ellRes3 = galSBP.galSBP(imgFile, mask=mskFile,
                                    galX=galX0,
                                    galY=galY0,
                                    maxSma=maxR,
                                    iniSma=iniSma,
                                    galR=galR50,
                                    ellipStep=step,
                                    maxTry=maxTry,
                                    pix=pix,
                                    bkg=bkg,
                                    galQ=galQ0,
                                    galPA=galPA0,
                                    stage=3,
                                    zpPhoto=zp,
                                    updateIntens=updateIntens,
                                    olthresh=olthresh,
                                    lowClip=lowClip,
                                    uppClip=uppClip,
                                    nClip=nClip,
                                    fracBad=fracBad,
                                    intMode=intMode,
                                    plMask=plMask,
                                    suffix=suffix,
                                    imgType=imgType,
                                    isophote=isophote,
                                    xttools=xttools)
            ellOut3, ellBin3 = ellRes3

            if ellOut3 is None:
                raise Exception("!!!!! ELLIPSE RUN FAILED AT STAGE 3 !!!!")
            else:
                """
                Should update the maxR
                """
                maxR = np.nanmax(ellOut3['sma'])

            if plot:
                if suffix[-1] != '_':
                    suffix = suffix + '_'
                if imgSub:
                    temp = '_imgsub_ellip_'
                    sumPng = root + prefix + temp + suffix + 'sum.png'
                else:
                    temp = '_img_ellip_'
                    sumPng = root + prefix + temp + suffix + 'sum.png'
                try:
                    ellipSummary(ellOut1, ellOut2, ellOut3, imgOri,
                                 psfOut=psfOut,
                                 maxRad=maxR, mask=mskOri, radMode='rsma',
                                 outPng=sumPng, zp=zp, useKpc=useKpc, pix=pix,
                                 showZoom=showZoom, exptime=exptime, bkg=bkg,
                                 outRatio=outRatio, imgType=imgType,
                                 verbose=verbose)
                except Exception:
                    print("XXX    Can not make summary plot: %s" % sumPng)

            if multiEllipse:
                """
                Run Ellipse using different mask and configuration

                This is mainly to test the robustness of the 1-D SBP
                """

                """ 1. Small Mask """
                if not noMask:
                    mskSmall = mskFile.replace('mskfin', 'msksmall')
                    inputSmall = ellBin3
                    suffixSmall = 'multi1'
                    if verbose:
                        print("\n###    Force Ellipse Run with Small Mask ")
                        print("###    Mask / Input : %s / %s" % (mskSmall,
                                                                 inputSmall))
                    smallOut = galSBP.galSBP(imgFile, mask=mskSmall,
                                             ellipStep=step,
                                             stage=4,
                                             zpPhoto=zp,
                                             pix=pix,
                                             bkg=bkg,
                                             maxTry=1,
                                             updateIntens=updateIntens,
                                             inEllip=inputSmall,
                                             suffix=suffixSmall,
                                             imgType=imgType,
                                             isophote=isophote,
                                             xttools=xttools)
                    smallEll, smallBin = smallOut
                else:
                    smallEll = None

                """ 2. Large Mask """
                if not noMask:
                    mskLarge = mskFile.replace('mskfin', 'msklarge')
                    inputLarge = ellBin3
                    suffixLarge = 'multi2'
                    if verbose:
                        print("\n###    Force Ellipse Run with Large Mask ")
                        print("###    Mask / Input : %s / %s" % (mskLarge,
                                                                 inputLarge))
                    largeOut = galSBP.galSBP(imgFile, mask=mskLarge,
                                             ellipStep=step,
                                             stage=4,
                                             zpPhoto=zp,
                                             pix=pix,
                                             bkg=bkg,
                                             maxTry=1,
                                             updateIntens=updateIntens,
                                             inEllip=inputLarge,
                                             suffix=suffixLarge,
                                             imgType=imgType,
                                             isophote=isophote,
                                             xttools=xttools)
                    largeEll, largeBin = largeOut
                else:
                    largeEll = None

                """ 3. Strick clipping """
                suffixMulti3 = 'multi3'
                if verbose:
                    print("\n###    Ellipse Run wit Strick Pixel Clipping ")
                resMulti3 = galSBP.galSBP(imgFile, mask=mskFile,
                                          galX=galX0,
                                          galY=galY0,
                                          maxSma=maxR,
                                          iniSma=iniSma,
                                          galR=galR50,
                                          ellipStep=step,
                                          maxTry=maxTry,
                                          pix=pix,
                                          bkg=bkg,
                                          galQ=galQ0,
                                          galPA=galPA0,
                                          zpPhoto=zp,
                                          updateIntens=updateIntens,
                                          olthresh=olthresh,
                                          stage=3,
                                          lowClip=2.2,
                                          uppClip=2.2,
                                          nClip=3,
                                          fracBad=0.5,
                                          intMode=intMode,
                                          plMask=plMask,
                                          suffix=suffixMulti3,
                                          imgType=imgType,
                                          isophote=isophote,
                                          xttools=xttools)
                ellMulti3, binMulti3 = resMulti3

                """ 4. Larger step size """
                suffixMulti4 = 'multi4'
                if verbose:
                    print("\n###    Ellipse Run wit Strick Pixel Clipping ")
                resMulti4 = galSBP.galSBP(imgFile, mask=mskFile,
                                          galX=galX0,
                                          galY=galY0,
                                          maxSma=maxR,
                                          iniSma=iniSma,
                                          galR=galR50,
                                          ellipStep=0.24,
                                          maxTry=maxTry,
                                          pix=pix,
                                          bkg=bkg,
                                          galQ=galQ0,
                                          galPA=galPA0,
                                          zpPhoto=zp,
                                          updateIntens=updateIntens,
                                          olthresh=olthresh,
                                          stage=3,
                                          lowClip=lowClip,
                                          uppClip=uppClip,
                                          nClip=nClip,
                                          fracBad=fracBad,
                                          intMode=intMode,
                                          plMask=plMask,
                                          suffix=suffixMulti4,
                                          imgType=imgType,
                                          isophote=isophote,
                                          xttools=xttools)
                ellMulti4, binMulti4 = resMulti4

                """ 5. Use Mean instead of Median """
                suffixMulti5 = 'multi5'
                if verbose:
                    print("\n###    Ellipse Run with Median integration mode ")
                resMulti5 = galSBP.galSBP(imgFile, mask=mskFile,
                                          galX=galX0,
                                          galY=galY0,
                                          maxSma=maxR,
                                          iniSma=iniSma,
                                          galR=galR50,
                                          ellipStep=step,
                                          maxTry=maxTry,
                                          pix=pix,
                                          bkg=bkg,
                                          galQ=galQ0,
                                          galPA=galPA0,
                                          zpPhoto=zp,
                                          updateIntens=updateIntens,
                                          olthresh=olthresh,
                                          stage=3,
                                          lowClip=lowClip,
                                          uppClip=uppClip,
                                          nClip=nClip,
                                          fracBad=fracBad,
                                          intMode='mean',
                                          plMask=plMask,
                                          suffix=suffixMulti5,
                                          imgType=imgType,
                                          isophote=isophote,
                                          xttools=xttools)
                ellMulti5, binMulti5 = resMulti5

                """ """
                ellStack = [ellOut3, smallEll, largeEll,
                            ellMulti3, ellMulti4, ellMulti5]
                ellLabel = ['Standard', 'Small Mask',
                            'Large Mask',
                            'Large nClip',
                            'Large Step',
                            'intMode=Mean']
                if imgSub:
                    comparePng = (root + prefix + '_imgsub_ellip_' + suffix +
                                  'compare.png')
                else:
                    comparePng = (root + prefix + '_img_ellip_' + suffix +
                                  'compare.png')
                ellipCompare(ellStack, outPng=comparePng, zp=27.0,
                             ellipLabel=ellLabel)
            else:
                smallEll, largeEll = None, None
                ellMulti3, ellMulti4, ellMulti5 = None, None, None
        else:
            """ # Run Ellipse in Forced Photometry Mode """
            if verbose:
                print("\n###    Ellipse Run on " +
                      "Image %s - Forced Photometry " % imgFile)
            ellOut4 = galSBP.galSBP(imgFile, mask=mskFile,
                                    galX=galX,
                                    galY=galY,
                                    inEllip=inEllip,
                                    pix=pix,
                                    bkg=bkg,
                                    stage=4,
                                    zpPhoto=zp,
                                    maxTry=1,
                                    updateIntens=updateIntens,
                                    intMode=intMode,
                                    plMask=plMask,
                                    suffix=suffix,
                                    imgType=imgType,
                                    isophote=isophote,
                                    xttools=xttools)
            if ellOut4 is None:
                raise Exception("!!!!! FORCED ELLIPSE RUN FAILED !!!!")

            gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Prefix of the input image")
    parser.add_argument('-r', "--root", dest='root',
                        help="Directory for the input image",
                        default=None)
    parser.add_argument("--intMode", dest='intMode',
                        help="Method for integration",
                        default='mean')
    parser.add_argument('--inEllip', dest='inEllip',
                        help='Input Ellipse table',
                        default=None)
    "" "Optional """
    parser.add_argument("--suffix",
                        help="Suffix of the output image",
                        default='')
    parser.add_argument('--exMask',
                        help="External file for image mask",
                        default=None)
    parser.add_argument('--pix', dest='pix',
                        help='Pixel Scale',
                        type=float, default=0.168)
    parser.add_argument('--step', dest='step',
                        help='Step size',
                        type=float, default=0.12)
    parser.add_argument('--zp', dest='zp',
                        help='Photometric zeropoint',
                        type=float, default=27.0)
    parser.add_argument('--redshift', dest='redshift',
                        help='Photometric zeropoint',
                        type=float, default=None)
    parser.add_argument('--olthresh', dest='olthresh',
                        help='Central locator threshold',
                        type=float, default=0.50)
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
    parser.add_argument('--galY0', dest='galY0',
                        help='Center Y0',
                        type=float, default=None)
    parser.add_argument('--galQ0', dest='galQ0',
                        help='Input Axis Ratio',
                        type=float, default=None)
    parser.add_argument('--galPA0', dest='galPA0',
                        help='Input Position Angle',
                        type=float, default=None)
    parser.add_argument('--galRe', dest='galRe',
                        help='Input Effective Radius in pixel',
                        type=float, default=None)
    parser.add_argument('--outRatio', dest='outRatio',
                        help='Increase the outer boundary by this ratio',
                        type=float, default=1.2)
    parser.add_argument('--verbose', dest='verbose', action="store_true",
                        default=True)
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
                        action="store_true", default=True)
    parser.add_argument('--multiEllipse', dest='multiEllipse',
                        action="store_true",
                        default=False)
    parser.add_argument('--plmask', dest='plmask', action="store_true",
                        default=True)
    parser.add_argument('--nomask', dest='nomask', action="store_true",
                        default=False)
    parser.add_argument('--imgSub', dest='imgSub', action="store_true",
                        default=False)
    parser.add_argument("--isophote", dest='isophote',
                        help="Location of the x_isophote.e file",
                        default=None)
    parser.add_argument("--xttools", dest='xttools',
                        help="Location of the x_ttools.e file",
                        default=None)

    args = parser.parse_args()

    coaddCutoutSbp(args.prefix, root=args.root,
                   verbose=args.verbose,
                   suffix=args.suffix,
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
                   plMask=args.plmask,
                   exMask=args.exMask,
                   noMask=args.noMask,
                   multiEllipse=args.multiEllipse,
                   imgSub=args.imgSub,
                   isophote=args.isophote,
                   xttools=args.xttools)
