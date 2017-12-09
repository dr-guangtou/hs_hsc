#!/usr/bin/env python
# encoding: utf-8
"""Prepare HSC cutout for photometry."""

from __future__ import (division, print_function)

import os
import copy
import glob
import argparse

import numpy as np

from distutils.version import StrictVersion

# Astropy
from astropy.wcs import WCS
from astropy.io import fits

# Scipy
import scipy.ndimage as ndimage

# SEP
import sep

# Personal
import hscUtils as hUtil
import ds9Reg2Mask as reg2Mask

# Matplotlib related
import matplotlib as mpl
from matplotlib.patches import Ellipse
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

sepVersion = sep.__version__
if StrictVersion(sepVersion) < StrictVersion('0.5.0'):
    raise Exception(
        'XXX SEP Version should be higher than 0.5.0; \n Please Update!!')

# For image
cmap1 = plt.get_cmap('viridis')
cmap1.set_bad('k', 1.)

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def seg2Mask(seg, sigma=5.0, mskMax=1000.0, mskThr=0.01):
    """
    Convert the segmentation array into an array.

    Parameters:
        sigma:  Sigma of the Gaussian Kernel
    """
    # Convolve the mask image with a gaussian kernel
    mskConv = ndimage.gaussian_filter(((seg.copy() > 0) * mskMax),
                                      sigma=sigma, order=0)
    mskBool = mskConv > (mskThr * mskMax)

    return mskBool.astype('uint8')


def objToGalfit(objs,
                rad=None,
                concen=None,
                zp=27.0,
                rbox=8.0,
                dimX=None,
                dimY=None):
    """
    objToGalfit.

    Empirical code that convert the SEP parameters of detected objects
    into initial guess of parameters for 1-Sersic GALFIT fit.
    """
    """ Number of objects """
    nObj = len(objs)
    """ Define a dictionary includes the necessary information """
    galfit1C = np.recarray(
        (nObj, ),
        dtype=[('xcen', float), ('ycen', float), ('mag', float), ('re', float),
               ('n', float), ('ba', float), ('pa', float), ('lowX', int),
               ('lowY', int), ('uppX', int), ('uppY', int), ('truncate',
                                                             bool), ('small',
                                                                     bool)])
    """ Effective radius used for GALFIT """
    if rad is None:
        rad = objs['a'] * 3.0
    for ii, obj in enumerate(objs):
        galfit1C[ii]['xcen'] = obj['x']
        galfit1C[ii]['ycen'] = obj['y']
        galfit1C[ii]['mag'] = -2.5 * np.log10(obj['cflux'] * 2.5) + zp
        galfit1C[ii]['re'] = rad[ii]
        galfit1C[ii]['ba'] = (obj['b'] / obj['a'])
        galfit1C[ii]['pa'] = (obj['theta'] * 180.0 / np.pi)
        # Initial guess for the Sersic index
        # Make it as simple as possible at first
        if concen is not None:
            if concen[ii] > 2.2:
                galfit1C[ii]['n'] = 2.5
            else:
                galfit1C[ii]['n'] = 1.0
        else:
            galfit1C[ii]['n'] = 1.5
        # Define a fitting box
        lowX = int(obj['x'] - rbox * rad[ii])
        lowY = int(obj['y'] - rbox * rad[ii])
        uppX = int(obj['x'] + rbox * rad[ii])
        uppY = int(obj['y'] + rbox * rad[ii])
        galfit1C[ii]['truncate'] = False
        small = 0
        if lowX < 0:
            lowX = 0
            galfit1C[ii]['truncate'] = True
            small += 1
        if lowY < 0:
            lowY = 0
            galfit1C[ii]['truncate'] = True
            small += 1
        if (dimX is not None) and (uppX > dimX):
            uppX = dimX
            galfit1C[ii]['truncate'] = True
            small += 1
        if (dimY is not None) and (uppY > dimY):
            uppY = dimY
            galfit1C[ii]['truncate'] = True
            small += 1
        if small >= 3:
            galfit1C[ii]['small'] = True
        galfit1C[ii]['lowX'] = lowX
        galfit1C[ii]['lowY'] = lowY
        galfit1C[ii]['uppX'] = uppX
        galfit1C[ii]['uppY'] = uppY

    return galfit1C


def showObjects(objs,
                dist,
                rad=None,
                outPNG='sep_object.png',
                cenInd=None,
                prefix=None,
                r1=None,
                r2=None,
                r3=None,
                fluxRatio1=None,
                fluxRatio2=None,
                highlight=None):
    """
    Plot the properties of objects detected on the images.

    Parameters:
    """
    fontsize = 13
    # Choice of radius to plot
    if rad is not None:
        r = rad
    else:
        r = objs['a']
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    fig.subplots_adjust(
        hspace=0.18, wspace=0.28, left=0.10, bottom=0.07, top=0.99, right=0.99)
    #  Fig1
    if fluxRatio1 is not None:
        axes[0, 0].axvline(
            np.log10(objs[cenInd]['flux'] * fluxRatio1),
            linestyle='--',
            lw=2.0)
    if fluxRatio2 is not None:
        axes[0, 0].axvline(
            np.log10(objs[cenInd]['flux'] * fluxRatio2), linestyle=':', lw=2.0)
    axes[0, 0].scatter(
        np.log10(objs['flux']),
        np.log10(r),
        facecolors='g',
        edgecolors='gray',
        alpha=0.30,
        s=(500.0 / np.sqrt(dist)))
    axes[0, 0].set_xlabel('log(Flux)', fontsize=15)
    axes[0, 0].set_ylabel('log(Radius/pixel)', fontsize=15)
    if highlight is not None:
        axes[0, 0].scatter(
            np.log10(objs[highlight]['flux']),
            np.log10(r[highlight]),
            color='b',
            alpha=0.50,
            s=(500.0 / np.sqrt(dist[highlight])))
    if cenInd is not None:
        axes[0, 0].scatter(
            np.log10(objs[cenInd]['flux']),
            np.log10(r[cenInd]),
            color='r',
            alpha=0.60,
            s=(500.0 / np.sqrt(dist[cenInd])))
    axes[0, 0].text(
        0.60,
        0.06,
        'Size: Approximity',
        fontsize=16,
        transform=axes[0, 0].transAxes,
        ha='center')

    #  Fig2
    axes[0, 1].scatter(
        np.log10(objs['flux'] / (objs['a'] * objs['b'])),
        np.log10(r),
        facecolors='g',
        edgecolors='gray',
        alpha=0.30,
        s=(500.0 / np.sqrt(dist)))
    axes[0, 1].set_xlabel('log(Flux/Area)', fontsize=15)
    axes[0, 1].set_ylabel('log(Radius/pixel)', fontsize=15)
    if highlight is not None:
        axes[0, 1].scatter(
            (np.log10(objs[highlight]['flux'] /
                      (objs[highlight]['a'] * objs[highlight]['b']))),
            np.log10(r[highlight]),
            color='b',
            alpha=0.50,
            s=(500.0 / np.sqrt(dist[highlight])))
    if cenInd is not None:
        axes[0, 1].scatter(
            (np.log10(objs[cenInd]['flux'] /
                      (objs[cenInd]['a'] * objs[cenInd]['b']))),
            np.log10(r[cenInd]),
            color='r',
            alpha=0.60,
            s=(500.0 / np.sqrt(dist[cenInd])))
    axes[0, 1].text(
        0.60,
        0.06,
        'Size: Approximity',
        fontsize=16,
        transform=axes[0, 1].transAxes,
        ha='center')
    if prefix is not None:
        axes[0, 1].text(
            0.50,
            0.91,
            prefix,
            fontsize=15,
            ha='center',
            transform=axes[0, 0].transAxes)

    #  Fig3
    if fluxRatio1 is not None:
        axes[1, 0].axhline(
            np.log10(objs[cenInd]['flux'] * fluxRatio1),
            linestyle='--',
            lw=2.0)
    if fluxRatio2 is not None:
        axes[1, 0].axhline(
            np.log10(objs[cenInd]['flux'] * fluxRatio2), linestyle=':', lw=2.0)
    if r1 is not None:
        axes[1, 0].axvline(r1, linestyle='-', lw=2.0)
    if r2 is not None:
        axes[1, 0].axvline(r2, linestyle='--', lw=2.0)
    if r3 is not None:
        axes[1, 0].axvline(r3, linestyle=':', lw=2.0)
    axes[1, 0].scatter(
        dist,
        np.log10(objs['flux']),
        facecolors='g',
        edgecolors='gray',
        alpha=0.30,
        s=(r * 10.0))
    axes[1, 0].set_xlabel('Central Distance (pixels)', fontsize=15)
    axes[1, 0].set_ylabel('log(Flux)', fontsize=15)
    if highlight is not None:
        axes[1, 0].scatter(
            dist[highlight],
            np.log10(objs[highlight]['flux']),
            color='b',
            alpha=0.50,
            s=(r[highlight] * 35.0))
    if cenInd is not None:
        axes[1, 0].scatter(
            dist[cenInd],
            np.log10(objs[cenInd]['flux']),
            color='r',
            alpha=0.60,
            s=(r[cenInd] * 35.0))
    axes[1, 0].text(
        0.60,
        0.06,
        'Size: Object Size',
        fontsize=16,
        transform=axes[1, 0].transAxes,
        ha='center')

    #  Fig4
    if r1 is not None:
        axes[1, 1].axvline(r1, linestyle='-', lw=2.0)
    if r2 is not None:
        axes[1, 1].axvline(r2, linestyle='--', lw=2.0)
    if r3 is not None:
        axes[1, 1].axvline(r3, linestyle=':', lw=2.0)
    axes[1, 1].scatter(
        dist,
        np.log10(r),
        facecolors='g',
        edgecolors='gray',
        alpha=0.30,
        s=(np.log10(objs['flux'])**4.0 + 30.0))
    axes[1, 1].set_xlabel('Central Distance (pixels)', fontsize=15)
    axes[1, 1].set_ylabel('log(Radius/pixel)', fontsize=15)
    if highlight is not None:
        axes[1, 1].scatter(
            dist[highlight],
            np.log10(r[highlight]),
            color='b',
            alpha=0.50,
            s=(np.log10(objs[highlight]['flux'])**4.0 + 30.0))
    if cenInd is not None:
        axes[1, 1].scatter(
            dist[cenInd],
            np.log10(r[cenInd]),
            color='r',
            alpha=0.60,
            s=(np.log10(objs[cenInd]['flux'])**4.0 + 30.0))
    axes[1, 1].text(
        0.60,
        0.06,
        'Size: Object Flux',
        fontsize=16,
        transform=axes[1, 1].transAxes,
        ha='center')

    # Adjust the figure
    for ax in axes.flatten():
        ax.minorticks_on()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)

    # Save the figure
    fig.savefig(outPNG, dpi=60)
    plt.close(fig)


def showSEPImage(image,
                 contrast=0.2,
                 size=5,
                 cmap=cmap1,
                 title='Image',
                 pngName='sep.png',
                 titleInside=True,
                 ellList1=None,
                 ellList2=None,
                 ellList3=None,
                 ellList4=None,
                 ellColor1='b',
                 ellColor2='r',
                 ellColor3='g',
                 ell1=None,
                 ell2=None,
                 ell3=None,
                 ell4=None,
                 ellColor4='k',
                 ax=None,
                 mask=None,
                 mskAlpha=0.4):
    """
    Visualization of the results.

    Parameters:
    """
    fig = plt.figure(figsize=(size, size))
    fig.subplots_adjust(
        hspace=0.0, wspace=0.0, bottom=0.0, left=0.0, top=1.0, right=1.0)
    ax = fig.add_axes([0.002, 0.002, 0.998, 0.998])

    ax.set_title(title, fontsize=16, fontweight='bold', color='w')
    if not titleInside:
        ax.title.set_position((0.5, 1.01))
    else:
        ax.title.set_position((0.5, 0.90))

    imcopy = copy.deepcopy(image)
    imin, imax = hUtil.zscale(imcopy, contrast=contrast, samples=500)

    ax.imshow(
        np.arcsinh(imcopy),
        interpolation="none",
        vmin=imin,
        vmax=imax,
        cmap=cmap,
        origin='lower')

    if mask is not None:
        # imcopy[mask > 0] = np.nan
        ax.imshow(
            mask,
            interpolation="none",
            vmin=0,
            vmax=1,
            origin='lower',
            alpha=mskAlpha,
            cmap='gray_r')

    if ellList1 is not None:
        for e in ellList1:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_edgecolor(ellColor1)
            e.set_facecolor('none')
            e.set_linewidth(1.5)

    if ellList2 is not None:
        for e in ellList2:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_edgecolor(ellColor2)
            e.set_facecolor('none')
            e.set_linewidth(1.5)

    if ellList3 is not None:
        for e in ellList3:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_edgecolor(ellColor3)
            e.set_facecolor('none')
            e.set_linewidth(1.5)

    if ellList4 is not None:
        for e in ellList4:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_edgecolor('c')
            e.set_facecolor('none')
            e.set_linewidth(2.0)

    if ell1 is not None:
        ax.add_artist(ell1)
        ell1.set_clip_box(ax.bbox)
        ell1.set_alpha(0.8)
        ell1.set_edgecolor('r')
        ell1.set_facecolor('none')
        ell1.set_linewidth(2.0)
        ell1.set_linestyle('dashed')

    if ell2 is not None:
        ax.add_artist(ell2)
        ell2.set_clip_box(ax.bbox)
        ell2.set_alpha(0.8)
        ell2.set_edgecolor(ellColor4)
        ell2.set_facecolor('none')
        ell2.set_linewidth(2.5)
        ell2.set_linestyle('dashed')

    if ell3 is not None:
        ax.add_artist(ell3)
        ell3.set_clip_box(ax.bbox)
        ell3.set_alpha(0.8)
        ell3.set_edgecolor(ellColor4)
        ell3.set_facecolor('none')
        ell3.set_linewidth(2.5)
        ell3.set_linestyle('dashed')

    fig.savefig(pngName, dpi=90)
    plt.close(fig)


def addFlag(dictName, flagName, flagValue):
    """
    Add a flag to a dictionary.

    Parameters:
    """
    # If the dictionary for flags has not been defined, make a new one
    try:
        dictName
    except NameError:
        dictName = np.array([], dtype=[('name', 'a20'), ('value', 'i8')])
    # Assign a new flag
    newFlag = (str(flagName), flagValue)
    newDict = np.insert(dictName, 0, newFlag)

    return newDict

def obj2EllStr(obj, reg):
    """
    Generate a string to represent the elliptical shape of an object.
    """
    reg.write('ellipse ' + str(obj['x']) + ' ' + str(obj['y']) + ' ' +
              str(obj['a']) + ' ' + str(obj['b']) + ' ' +
              str(obj['theta'] * 180.0 / np.pi) + '\n')


def objList2Reg(objs, regName='ds9.reg', color='Blue'):
    """
    Save the Object List to DS9 Region File.

    Parameters:
    """
    # Open the output file and write the header
    regFile = open(regName, 'w')
    regFile.write('# Region file format: DS9 version 4.1\n')
    regFile.write('global color=%s width=2\n' % color)
    regFile.write('image\n')

    # Save the data
    # The format for ellipse is: "ellipse x y radius radius angle"
    [obj2EllStr(obj, regFile) for obj in objs]

    # Close the file
    regFile.close()


def saveFits(img, fitsName, head=None, overwrite=True):
    """
    Save an image to FITS file.

    Parameters:
    """
    imgHdu = fits.PrimaryHDU(img)
    if head is not None:
        imgHdu.header = head
    if os.path.islink(fitsName):
        os.unlink(fitsName)
    imgHdu.writeto(fitsName, overwrite=overwrite)


def saveSEPObjects(objs,
                   prefix='sep_objects',
                   csv=False,
                   reg=True,
                   verbose=False,
                   color='red',
                   pkl=False):
    """
    Name: saveSEPObjects.

    Save the properties of objects that are extracted by SEP into
    a cPickle file or .csv file, .deg file

    Parameters:
    """
    # 1. Save a .pkl file
    if pkl:
        pklFile = prefix + '.pkl'
        hUtil.saveToPickle(objs, pklFile)
        if os.path.isfile(pklFile):
            if verbose:
                print("\n###     Save object list to .pkl file: %s" % pklFile)
        else:
            raise Exception("### Something is wrong with the .pkl file")

    # 2. Save a .csv file
    if csv:
        csvFile = prefix + '.csv'
        hUtil.saveToCSV(objs, csvFile)
        if os.path.isfile(csvFile):
            if verbose:
                print("\n###     Save object list to .csv file: %s" % csvFile)
        else:
            raise Exception("### Something is wrong with the .csv file")

    # 3. Save a .reg file
    if reg:
        regFile = prefix + '.reg'
        objList2Reg(objs, regName=regFile, color=color)
        if os.path.isfile(regFile):
            if verbose:
                print("\n###     Save object list to .reg file: %s" % regFile)
        else:
            raise Exception("### Something is wrong with the .reg file")


def adaptiveMask(objC,
                 a=2.0,
                 b=1.5,
                 c=4.0,
                 seeing=1.0,
                 pix=0.168,
                 verbose=False):
    """
    Name: adaptiveMask.

    Scale the size of mask for different objects according to their
    flux and major axis radii

    XXX This is still very initial

    We adopt the form of:

    logRho = log10(Flux / (A * B))
    thrA = log10(seeing_fwhm / (2.0 * pixscale))
    Ratio = a * ((log(A) - thrA) <= 0 ? 0) + b * logRho + c
    """
    # Corresponding to the pixel size of a seeing disk
    # Use as a threshold to select "point-ish" objects
    thrA = np.log10(seeing / (2.0 * pix))
    # "Flux density" like properties
    logRho = np.log10(objC['cflux'] / (objC['a'] * objC['b']))
    # Logrithmic difference between the major axis radii and
    # the threshold size
    logA = (np.log10(np.sqrt(objC['a'])) - thrA)
    # Ignore all the "point-ish" objects
    logA[(logA < 0) | (logRho < 1.5)] = 0
    # Empirical function to get the mask size ratio
    rAdRatio = logA * a + logRho * b + c
    if verbose:
        print("### rAdRatio min, max, median : \
                %5.2f, %5.2f, %5.2f " %
              (np.nanmin(rAdRatio), np.nanmax(rAdRatio), np.median(rAdRatio)))
    return rAdRatio


def combMskImage(msk1, msk2):
    """
    Combine two mask images.

    Parameters:
    """
    if (msk1.shape[0] != msk2.shape[0]) or (msk1.shape[1] != msk2.shape[1]):
        raise Exception("### The two masks need to have the same shape!")
    return ((msk1 > 0) | (msk2 > 0))


def combObjCat(objCold, objHot, tol=6.0, cenDistC=None, keepH=True):
    """
    Merge the object lists from cold and hot run.

    tol = 6.0  : Tolerance of central difference in unit of pixel
    """
    # Make a copy of the objects array
    objC = copy.deepcopy(objCold)
    objH = copy.deepcopy(objHot)
    # The central coordinates of each objects
    objCX, objCY = objC['x'], objC['y']
    objHX, objHY = objH['x'], objH['y']
    # Get the minimum separation between each Hot run object and all the
    # Cold run objects
    minDistH = np.asarray(
        map(lambda x, y: np.min(np.sqrt((x - objCX)**2.0 + (y - objCY)**2.0)),
            objHX, objHY))
    minDistC = np.asarray(
        map(lambda x, y: np.min(np.sqrt((x - objHX)**2.0 + (y - objHY)**2.0)),
            objCX, objCY))
    # Locate the matched objects
    indMatchH = np.where(minDistH < tol)
    indMatchC = np.where(minDistC < tol)
    # Delete the matched objects from the Hot run list
    objHnew = copy.deepcopy(objH)
    objHnew = np.delete(objHnew, indMatchH)
    objCnew = copy.deepcopy(objC)
    objCnew = np.delete(objCnew, indMatchC)

    if keepH:
        objComb = np.concatenate((objH, objCnew))
    else:
        objComb = np.concatenate((objC, objHnew))

    return objComb, objHnew, objCnew


def getConvKernel(kernel):
    """
    Convolution kernel for the SEP detections.

    Parameters:
    """
    if kernel is 1:
        # Tophat_3.0_3x3
        convKer = np.asarray([[0.560000, 0.980000,
                               0.560000], [0.980000, 1.000000, 0.980000],
                              [0.560000, 0.980000, 0.560000]])
    elif kernel is 2:
        # Topcat_4.0_5x5
        convKer = np.asarray(
            [[0.000000, 0.220000, 0.480000, 0.220000,
              0.000000], [0.220000, 0.990000, 1.000000, 0.990000, 0.220000],
             [0.480000, 1.000000, 1.000000, 1.000000,
              0.480000], [0.220000, 0.990000, 1.000000, 0.990000, 0.220000],
             [0.000000, 0.220000, 0.480000, 0.220000, 0.000000]])
    elif kernel is 3:
        # Topcat_5.0_5x5
        convKer = np.asarray(
            [[0.150000, 0.770000, 1.000000, 0.770000,
              0.150000], [0.770000, 1.000000, 1.000000, 1.000000, 0.770000],
             [1.000000, 1.000000, 1.000000, 1.000000,
              1.000000], [0.770000, 1.000000, 1.000000, 1.000000, 0.770000],
             [0.150000, 0.770000, 1.000000, 0.770000, 0.150000]])
    elif kernel is 4:
        # Gaussian_3.0_5x5
        convKer = np.asarray(
            [[0.092163, 0.221178, 0.296069, 0.221178,
              0.092163], [0.221178, 0.530797, 0.710525, 0.530797, 0.221178],
             [0.296069, 0.710525, 0.951108, 0.710525,
              0.296069], [0.221178, 0.530797, 0.710525, 0.530797, 0.221178],
             [0.092163, 0.221178, 0.296069, 0.221178, 0.092163]])
    elif kernel is 5:
        # Gaussian_4.0_7x7
        convKer = np.asarray([[
            0.047454, 0.109799, 0.181612, 0.214776, 0.181612, 0.109799,
            0.047454
        ], [
            0.109799, 0.254053, 0.420215, 0.496950, 0.420215, 0.254053,
            0.109799
        ], [
            0.181612, 0.420215, 0.695055, 0.821978, 0.695055, 0.420215,
            0.181612
        ], [
            0.214776, 0.496950, 0.821978, 0.972079, 0.821978, 0.496950,
            0.214776
        ], [
            0.181612, 0.420215, 0.695055, 0.821978, 0.695055, 0.420215,
            0.181612
        ], [
            0.109799, 0.254053, 0.420215, 0.496950, 0.420215, 0.254053,
            0.109799
        ], [
            0.047454, 0.109799, 0.181612, 0.214776, 0.181612, 0.109799,
            0.047454
        ]])
    elif kernel is 6:
        # Gaussian_5.0_9x9
        convKer = np.asarray([[
            0.030531, 0.065238, 0.112208, 0.155356, 0.173152, 0.155356,
            0.112208, 0.065238, 0.030531
        ], [
            0.065238, 0.139399, 0.239763, 0.331961, 0.369987, 0.331961,
            0.239763, 0.139399, 0.065238
        ], [
            0.112208, 0.239763, 0.412386, 0.570963, 0.636368, 0.570963,
            0.412386, 0.239763, 0.112208
        ], [
            0.155356, 0.331961, 0.570963, 0.790520, 0.881075, 0.790520,
            0.570963, 0.331961, 0.155356
        ], [
            0.173152, 0.369987, 0.636368, 0.881075, 0.982004, 0.881075,
            0.636368, 0.369987, 0.173152
        ], [
            0.155356, 0.331961, 0.570963, 0.790520, 0.881075, 0.790520,
            0.570963, 0.331961, 0.155356
        ], [
            0.112208, 0.239763, 0.412386, 0.570963, 0.636368, 0.570963,
            0.412386, 0.239763, 0.112208
        ], [
            0.065238, 0.139399, 0.239763, 0.331961, 0.369987, 0.331961,
            0.239763, 0.139399, 0.065238
        ], [
            0.030531, 0.065238, 0.112208, 0.155356, 0.173152, 0.155356,
            0.112208, 0.065238, 0.030531
        ]])
    else:
        raise Exception("### More options will be available in the future")

    return convKer


def getEll2Plot(objects, radius=None):
    """
    Generate the ellipse shape for each object to plot.

    Parameters:
    """
    x = objects['x'].copy()
    y = objects['y'].copy()
    pa = objects['theta'].copy()  # in unit of radian

    if radius is not None:
        a = radius.copy()
        b = radius.copy() * (objects['b'].copy() / objects['a'].copy())
    else:
        a = objects['a'].copy()
        b = objects['b'].copy()

    return [Ellipse(xy=np.array([x[i], y[i]]),
                    width=np.array(2.0 * b[i]),
                    height=np.array(2.0 * a[i]),
                    angle=np.array(pa[i] * 180.0 / np.pi + 90.0))
            for i in range(x.shape[0])]


def getSbpValue(flux, pixX, pixY, zp=None):
    """
    Convert flux into surface brightness value.

    TODO: Right now only support log-magnitude,
    In the future, should also support asinh-magnitude
    See:
    http://www.astro.washington.edu/users/ajc/ssg_page_working/elsst/opsim.shtml?lightcurve_mags
    """
    sbp = -2.5 * np.log10(flux / (pixX * pixY))
    if zp is not None:
        sbp += zp
    return sbp


def getFluxRadius(img, objs, maxSize=6.0, subpix=5, byteswap=True, mask=None):
    """
    Name: getFluxRadius.

    Given the original image, the detected objects, using SEP
    to measure different flux radius: R20, R50, R90
    """
    imgOri = copy.deepcopy(img)
    if byteswap:
        imgOri = imgOri.byteswap(True).newbyteorder()
    else:
        imgOri = img
    # Get the flux radius
    if mask is not None:
        rflux, flag = sep.flux_radius(
            imgOri,
            objs['x'],
            objs['y'],
            maxSize * objs['a'], [0.2, 0.5, 0.9],
            normflux=objs['cflux'],
            subpix=subpix,
            mask=mask.astype('uint8'),
            maskthresh=0)
    else:
        rflux, flag = sep.flux_radius(
            imgOri,
            objs['x'],
            objs['y'],
            maxSize * objs['a'], [0.2, 0.5, 0.9],
            normflux=objs['flux'],
            subpix=subpix)
    if isinstance(objs['x'], (int, long, float)):
        r20, r50, r90 = rflux[0], rflux[1], rflux[2]
    else:
        r20 = np.array([rr[0] for rr in rflux])
        r50 = np.array([rr[1] for rr in rflux])
        r90 = np.array([rr[2] for rr in rflux])

    return r20, r50, r90


def objDistTo(objs,
              cenX,
              cenY,
              usePeak=False,
              convol=False,
              ellipse=True,
              pa=0.0,
              q=0.99):
    """
    Get the distance of objects from SEP to a reference point on the image.

    Parameters:
    """
    if usePeak:
        if convol:
            xc, yc = objs['xcpeak'], objs['ycpeak']
        else:
            xc, yc = objs['xpeak'], objs['ypeak']
    else:
        xc, yc = objs['x'], objs['y']

    if ellipse:
        theta = (pa * np.pi / 180.0)
        distA = ((xc - cenX) * np.cos(theta) +
                 (yc - cenY) * np.sin(theta))**2.0
        distB = (((yc - cenY) * np.cos(theta) -
                  (xc - cenX) * np.sin(theta)) / q)**2.0
        return np.sqrt(distA + distB)
    else:
        return np.sqrt((xc - cenX)**2 + (yc - cenY)**2)


def readCutoutHeader(imgHead, pixDefault=0.168, zpDefault=27.0):
    """
    Name: readCutoutHeader.

    Read the pixel scale, image size, and photometric zeropoint form
    the image header

    TODO: Make it more generic, right now it is only for HSC
     * pixel scale can be read from the WCS information
     * Right now, the TOTEXPT is not working for HSC
    """
    # Get the pixel scale of the image
    try:
        pixScaleX = pixScaleY = imgHead['PIXEL']
    except Exception:
        pixScaleX = pixScaleY = pixDefault
    # Get the image size
    imgSizeX = imgHead['NAXIS1']
    imgSizeY = imgHead['NAXIS2']

    # Get the photometric zeropoint
    try:
        photZP = imgHead['PHOTZP']
    except Exception:
        photZP = zpDefault

    # Total exptime
    try:
        expTot = imgHead['TOTEXPT']
    except Exception:
        expTot = 1.0

    return pixScaleX, pixScaleY, imgSizeX, imgSizeY, photZP, expTot


def imgByteSwap(data):
    """
    Byte Swap before sending image to SEP.

    Parameters:
    """
    # dataCopy = copy.deepcopy(data)
    return data.copy().byteswap(True).newbyteorder()


def sepValidObjects(obj):
    """
    Remove objects with negative flux or small size.
    """
    return obj[(obj['flux'] > 0.0) & (obj['a'] > 0.01)]


def sepGetBkg(img, mask=None, bkgSize=None, bkgFilter=None):
    """
    Wrapper of SEP.Background function.

    Parameters:
    """
    if bkgSize is None:
        dimX, dimY = img.shape
        bkgX = img(dimX / 15)
        bkgY = img(dimY / 15)
    else:
        bkgX = bkgY = int(bkgSize)
    if bkgFilter is None:
        bkgFilter = 4

    bkg = sep.Background(img, mask=mask, bw=bkgX, bh=bkgY,
                         fw=bkgFilter, fh=bkgFilter)
    # Subtract the Background off
    bkg.subfrom(img)

    return bkg, img


def readCutoutImage(prefix, root=None, variance=False):
    """
    Read the cutout images.

    Parameters:
    """
    # Get the names of necessary input images
    imgFile = prefix + '_img.fits'
    mskFile = prefix + '_bad.fits'
    detFile = prefix + '_det.fits'
    if not variance:
        sigFile = prefix + '_sig.fits'
    else:
        sigFile = prefix + '_var.fits'

    if root is not None:
        imgFile = os.path.join(root, imgFile)
        mskFile = os.path.join(root, mskFile)
        detFile = os.path.join(root, detFile)
        sigFile = os.path.join(root, sigFile)

    # Image Data
    if os.path.islink(imgFile):
        imgOri = os.readlink(imgFile)
        imgFile = imgOri

    if os.path.isfile(imgFile):
        imgHdu = fits.open(imgFile)
        imgArr = imgHdu[0].data
    else:
        raise Exception(
            "### Can not find the Input Image File : %s !" % imgFile)
    # Header
    imgHead = imgHdu[0].header

    # Bad mask
    if os.path.islink(mskFile):
        mskOri = os.readlink(mskFile)
        mskFile = mskOri
    if os.path.isfile(mskFile):
        mskArr = fits.open(mskFile)[0].data
    else:
        print("\n### Can not find the coadd BadPlane file!")
        mskArr = None

    # Optional detection plane
    if os.path.islink(detFile):
        detOri = os.readlink(detFile)
        detFile = detOri
    if os.path.isfile(detFile):
        detArr = fits.open(detFile)[0].data
    else:
        print("### Can not find the coadd DetectionPlane file!")
        detArr = None

    # Optional sigma plane
    if os.path.islink(sigFile):
        sigOri = os.readlink(sigFile)
        sigFile = sigOri
    if os.path.isfile(sigFile):
        sigArr = fits.open(sigFile)[0].data
    else:
        print("\n### Can not find the coadd sigectionPlane file!")
        sigArr = None

    return imgArr, imgHead, mskArr, detArr, sigArr


def matchStarCatalog(starCat, imgHead, margin=200, aggres=600):
    """
    Match the bright star catalog with the image.
    Return the X, Y positions of the matched stars and
    the radius of the masks.
    """
    dimX, dimY = imgHead['NAXIS1'], imgHead['NAXIS2']
    pixel = 0.168   # arcsec/ pixel

    imgWcs = WCS(imgHead)
    ra0, dec0 = imgWcs.all_pix2world(0 - margin,
                                     0 - margin,
                                     0)
    ra1, dec1 = imgWcs.all_pix2world(dimX + margin,
                                     dimY + margin,
                                     0)
    ra_min, ra_max = np.min([ra0, ra1]), np.max([ra0, ra1])
    dec_min, dec_max = np.min([dec0, dec1]), np.max([dec0, dec1])

    starClose = starCat[(starCat['ra'] >= ra_min) &
                        (starCat['ra'] <= ra_max) &
                        (starCat['dec'] >= dec_min) &
                        (starCat['dec'] <= dec_max)]

    if len(starClose) > 0:
        xStar, yStar = imgWcs.all_world2pix(starClose['ra'],
                                            starClose['dec'],
                                            0)
        rStar = (aggres * np.exp(-starClose['mag'] / 4.04) /
                 pixel)
        return xStar, yStar, rStar
    else:
        return None, None, None


def maskStars(msk, xStar, yStar, rStar):
    """
    Generating masks for bright stars.
    """
    if isinstance(msk, fits.header.Header):
        new_msk = np.zeros((msk['NAXIS1'], msk['NAXIS2']),
                           dtype='uint8')
    else:
        new_msk = msk.copy()

    sep.mask_ellipse(new_msk, xStar, yStar, rStar, rStar,
                     rStar * 0.0)

    return new_msk


def coaddCutoutPrepare(prefix,
                       root=None,
                       verbose=True,
                       bSizeH=8,
                       bSizeC=40,
                       thrH=2.2,
                       thrC=1.2,
                       maskMethod=1,
                       growMethod=1,
                       central=1,
                       kernel=4,
                       growC=6.0,
                       growW=4.0,
                       growH=2.0,
                       galX=None,
                       galY=None,
                       galR1=None,
                       galR2=None,
                       galR3=None,
                       galQ=None,
                       galPA=None,
                       visual=True,
                       suffix='',
                       combBad=True,
                       combDet=True,
                       noBkgC=False,
                       noBkgH=False,
                       minDetH=5.0,
                       minDetC=8.0,
                       debThrH=16.0,
                       debThrC=32.0,
                       debConH=0.00001,
                       debConC=0.0001,
                       useSigArr=True,
                       minCenDist=10.0,
                       rerun='default',
                       segment=True,
                       mskReg=None,
                       excludeReg=None,
                       tol=10.0,
                       regMask=None,
                       regKeep=None,
                       sigma=6.0,
                       sigthr=0.02,
                       showAll=False,
                       brightStar=None,
                       multiMask=False):
    # Prepare SEP
    sep.set_extract_pixstack(500000)

    # Get the bright star catalog
    if brightStar is not None:
        if isinstance(brightStar, np.ndarray):
            starCat = brightStar
        else:
            starCat = hUtil.getStarCatalog()
    else:
        starCat = None

    """
    The structure of the cutout has been changed.

    Now the cutout procedure
    will generate separated files for Image, Bad Mask, Detection Plane, and
    Variance (also Sigma) images.  Souce catalogs can also be made available.

    Right now, this new format is only available for the coaddImageCutFull()
    function; coaddImageCutout() will be modified later to also adopt this
    format

    0. Get necessary information
    """
    # Read the input cutout image
    imgArr, imgHead, mskArr, detArr, sigArr = readCutoutImage(prefix,
                                                              root=root)
    if root is None:
        root = ''
    if verbose:
        print(SEP)
        print("\n### DEAL WITH IMAGE : %s" % (prefix + '_img.fits'))

    # Set up a rerun
    rerunDir = os.path.join(root, rerun.strip())
    if not os.path.isdir(rerunDir):
        os.makedirs(rerunDir)

    # Link the necessary files to the rerun folder
    fitsList = glob.glob(root + '*.fits')
    for fitsFile in fitsList:
        seg = fitsFile.split('/')
        link = os.path.join(rerunDir, seg[-1])
        if (not os.path.islink(link)) and (not os.path.isfile(link)):
            os.symlink(fitsFile, link)

    # DETECTION and BAD array is optional
    detFound, badFound = (detArr is not None), (mskArr is not None)
    mskArr = seg2Mask(mskArr) if badFound else None

    # Setup up an array for the flags """
    sepFlags = np.array([], dtype=[('name', 'a20'),
                                   ('value', 'i8')])

    # Sometimes NaN pixels exist for the image.
    indImgNaN = np.isnan(imgArr)
    sepFlags = addFlag(sepFlags, 'NAN_PIX', np.sum(indImgNaN))

    # Get the information from the header
    pixX, pixY, dimX, dimY, photZP, expTot = readCutoutHeader(imgHead)

    # Center of the galaxy
    galX = (dimX / 2.0) if galX is None else galX
    galY = (dimX / 2.0) if galY is None else galY

    # Define the suffix for the files
    if (suffix is not '') and (suffix[-1] is not '_'):
        suffix = suffix + '_'

    # Region to be masked out
    if (regMask is not None) and os.path.isfile(regMask):
        if verbose:
            print("\n###  Load in regMask : %s" % regMask)
        extMask = reg2Mask.reg2Mask(
            imgArr, regMask, hdu=0, save=False, imgHead=imgHead)
    else:
        extMask = None

    # Region to be kept on the image
    if (regKeep is not None) and os.path.isfile(regKeep):
        if verbose:
            print("\n###  Load in regKeep : %s" % regKeep)
        extKeep = reg2Mask.reg2Mask(
            imgArr, regKeep, hdu=0, save=False, imgHead=imgHead)
    else:
        extKeep = None

    """
    1. Get the backgrounds

    Construct "background" images with different size and filters using SEP,
    and subtract these background off before extract objects

    The SEP detections will be run in two-modes:
        Cold: relative global background; low-detection threshold
        Hot:  very local background; median-detection threshold
    """
    # Cold Background Run
    bkgC, imgSubC = sepGetBkg(imgByteSwap(imgArr), bkgSize=bSizeC, bkgFilter=7)
    if noBkgC:
        imgSubC = imgByteSwap(imgArr)

    # Hot Background Run
    bkgH, imgSubH = sepGetBkg(imgByteSwap(imgArr), bkgSize=bSizeH, bkgFilter=3)
    if noBkgH:
        imgSubH = imgByteSwap(imgArr)

    """
    2. Object detections

    Use SEP to extract information of detected objects
    """
    # Get the convolution kernel
    convKerC = getConvKernel(kernel)

    if useSigArr and (sigArr is not None):
        errArr, filter_type = imgByteSwap(sigArr), 'matched'
        detThrC, detThrH = thrC, thrH
    else:
        errArr, filter_type = None, 'conv'
        detThrC, detThrH = thrC * bkgC.globalrms, thrH * bkgH.globalrms

    # Cold Detection Run
    try:
        objC, segC = sep.extract(imgSubC, detThrC, minarea=minDetC,
                                 deblend_nthresh=debThrC,
                                 deblend_cont=debConC,
                                 filter_kernel=convKerC,
                                 filter_type=filter_type,
                                 err=errArr,
                                 segmentation_map=True)
    except Exception:
        # Try it once more
        print("### Failed the first time, try it again...")
        objC, segC = sep.extract(imgSubC, (detThrC, + 2),
                                 minarea=minDetC,
                                 deblend_nthresh=(debThrC * 2),
                                 deblend_cont=(debConC / 2),
                                 filter_kernel=convKerC,
                                 filter_type=filter_type,
                                 err=errArr,
                                 segmentation_map=True)
    if verbose:
        print("### A. COLD DETECTION: %d objects" % len(objC['x']))

    # Clean the objects
    objC = sepValidObjects(objC)

    # Save objects list to different format of files
    prefixC = os.path.join(rerunDir, (prefix + '_' + suffix + 'objC'))
    if showAll:
        saveSEPObjects(objC, prefix=prefixC, color='Blue', csv=False,
                       pkl=False, reg=True)

    # Hot Detection Run
    # Convolution kernel for hot run
    convKerH = getConvKernel(2)
    try:
        objH, segH = sep.extract(imgSubH, detThrH, minarea=minDetH,
                                 deblend_nthresh=debThrH,
                                 deblend_cont=debConH,
                                 filter_kernel=convKerH,
                                 filter_type=filter_type,
                                 err=errArr,
                                 segmentation_map=True)
    except Exception:
        # Try it once more
        print("### Failed the first time, try it again...")
        objH, segH = sep.extract(imgSubH, (detThrH, + 2),
                                 minarea=minDetH,
                                 deblend_nthresh=(debThrH * 2),
                                 deblend_cont=(debConH / 2),
                                 filter_kernel=convKerH,
                                 filter_type=filter_type,
                                 err=errArr,
                                 segmentation_map=True)
    if verbose:
        print("### B.  HOT DETECTION: %d objects" % len(objH['x']))

    # Clean the objects
    objH = sepValidObjects(objH)

    # Deal with the very elongated HOT object
    flagEll = (objH['b'] / objH['a']) < 0.5
    objH['a'][flagEll] /= 5.0
    objH['b'][flagEll] /= 3.0

    # Save objects list to different format of files
    prefixH = os.path.join(rerunDir, (prefix + '_' + suffix + 'objH'))
    if showAll:
        saveSEPObjects(
            objH, prefix=prefixH, color='Red', csv=False, pkl=False, reg=True)

    # Calculate the object-galaxy center distance
    cenDistC = objDistTo(objC, galX, galY)
    cenDistH = objDistTo(objH, galX, galY)

    # Index for central galaxy
    cenObjIndexC = np.nanargmin(cenDistC)
    cenObjC = objC[cenObjIndexC]
    cenObjIndexH = np.nanargmin(cenDistH)
    cenObjH = objH[cenObjIndexH]

    # Basic properties of the central galaxy
    galFlux, galA = cenObjC['cflux'], cenObjC['a']
    galCenX, galCenY = cenObjC['x'], cenObjC['y']
    galQ, galPA = (cenObjH['b'] / cenObjH['a']), (cenObjH['theta'] *
                                                  180.0 / np.pi)
    galA = galA if np.isfinite(galA) else 10.0
    galCenX = galCenX if np.isfinite(galCenX) else galX
    galCenY = galCenY if np.isfinite(galCenY) else galY
    galQ = galQ if np.isfinite(galQ) else 0.95
    galPA = galPA if np.isfinite(galPA) else 0.0
    if verbose:
        print("###    r: %6.2f; b/a : %4.2f; PA : %5.1f" % (cenObjH['a'],
                                                            galQ, galPA))

    """
    The Basic Parameters of the Central Galaxy:

        galCenX, galCenY : Center of the galaxy
        galFlux : Flux of the central galaxy
        galPA, galQ : The position angle and axis ratio
        galR20, galR50, galR90 : 20%, 50%, and 90% flux radius

    NOTE:
        When R20 fails: Use the size of the detection
        When R50 fails: R50 = A * 1.5
        When R90 fails: R50 = A * 3.0
        - Those numbers are pretty random...
    """
    # Prepare a mask for flux radius
    segTemp = segH.copy()
    segTemp[segTemp != (cenObjIndexC + 1)] = 0
    galR20, galR50, galR90 = getFluxRadius(
        imgArr, objH[cenObjIndexH], maxSize=20.0, subpix=5, byteswap=True,
        mask=None)
    galR20 = galR20 if np.isfinite(galR20) else 15.0
    galR50 = galR50 if np.isfinite(galR50) else 30.0
    galR90 = galR90 if np.isfinite(galR90) else 50.0

    # Make a flag if the galR90 is larger than 1/3 of the image size
    sepFlags = addFlag(sepFlags, 'R90_BIG',
                       (galR90 > (dimX / 3.0) or galR90 > (dimY / 3.0)))

    """
    Define a series of radius for masking:
        - galR1 = galR90 * 2.0
        - galR2 = galR90 * 4.0
        - galR3 = galR90 * 6.0
        - These numbers are pretty random too..sorry
    """
    galR1 = (galR50 * 2.0) if galR1 is None else (galR1 * galR50)
    galR2 = (galR50 * 4.0) if galR2 is None else (galR2 * galR50)
    galR3 = (galR50 * 6.0) if galR3 is None else (galR3 * galR50)
    if verbose:
        print("###    R20/50/90 : %6.2f / %6.2f / %6.2f" % (
                galR20, galR50, galR90))
        print("###    galR1/R2/R3 : %6.2f / %6.2f / %6.2f" % (
                galR1, galR2, galR3))

    # Make a flag if the galR4 it larger than 2/3 of the image size
    sepFlags = addFlag(sepFlags, 'R3_BIG', (galR3 >= dimX / 1.5))

    # Define a region that encloses the entire galaxy
    mskGal = np.zeros(imgSubC.shape, dtype='uint8')
    mskR2 = np.zeros(imgSubC.shape, dtype='uint8')
    sep.mask_ellipse(mskGal, galX, galY, galR3, (galR3 * galQ),
                     (galPA * np.pi / 180.0), r=1.1)
    sep.mask_ellipse(mskR2, galX, galY, galR2, (galR2 * galQ),
                     (galPA * np.pi / 180.0), r=1.1)

    # Clear up the DETECTION mask plane in this region
    detMsk = seg2Mask(((detArr > 0) & ~mskGal),
                      sigma=3, mskThr=sigthr) if detArr is not None else None

    # Estimate the distance to the central galaxies in elliptical coordinates
    cenDistC = objDistTo(objC, galX, galY, pa=galPA, q=galQ)
    cenDistH = objDistTo(objH, galX, galY, pa=galPA, q=galQ)

    """
    3. Merge the objects from Cold and Hot runs together
    """
    objComb, objHnew, objCnew = combObjCat(
        objC, objH, keepH=True, cenDistC=cenDistC, tol=tol)

    # Also save the combined object lists
    prefixComb = os.path.join(rerunDir, (prefix + '_' + suffix + 'objComb'))
    saveSEPObjects(objComb, prefix=prefixComb, color='Green',
                   csv=False, pkl=False, reg=True)

    # Calculate the object-galaxy center distance...Again
    cenDistComb = objDistTo(objComb, galX, galY, pa=galPA, q=galQ)
    cenObjIndex = np.nanargmin(cenDistComb)

    if verbose:
        print("### C.  COMBINED DETECTION: %d objects" % len(objComb['x']))

    """
    4. Extract Different Flux Radius: R20, R50, R90 for every objects
    """
    r20, r50, r90 = getFluxRadius(
        imgByteSwap(imgSubH), objComb, maxSize=6.0, subpix=5)
    rPhoto = objComb['a']
    # Some objects at the edge could have failed R50/R90, replace them with:
    # a * factor;  factor is still pretty random
    r20[np.isnan(r20)] = rPhoto[np.isnan(r20)]
    r50[np.isnan(r50)] = rPhoto[np.isnan(r50)] * 3.0
    r90[np.isnan(r90)] = rPhoto[np.isnan(r90)] * 5.0

    # Deal with the oversized faint objects; this is a pretty rough way
    overSize = ((np.log10(objComb['flux']) < 2.5) &
                (np.log10(r90) > 1.2))
    print("###    Find %d oversize objects" % np.sum(overSize))
    r90[overSize] /= 5.0
    r50[overSize] /= 5.0
    r20[overSize] /= 5.0
    # Also make them rounder
    objComb['a'][overSize] /= 5.0
    objComb['b'][overSize] /= 2.5

    """
    5. Mask all objects on the image
    """
    objMskAll = copy.deepcopy(objComb)
    # Convolve the segmentations into a masks
    mskAll = np.zeros(imgSubC.shape, dtype='uint8')

    # By default, grow every object by "growC * 1.5"
    sep.mask_ellipse(mskAll, objMskAll['x'], objMskAll['y'],
                     objMskAll['a'], objMskAll['b'],
                     objMskAll['theta'], r=growH)
    mskAll = (mskAll | mskGal)
    mskAll[indImgNaN] = 1
    objMskAll['a'] *= growH
    objMskAll['b'] *= growH

    # Combined the all object mask with the BAD_MASK and DET_MASK from pipeline
    if combBad and badFound:
        mskAll = combMskImage(mskAll, mskArr)
    if combDet and detFound:
        detAll = copy.deepcopy(detArr).astype(int)
        detMskAll = seg2Mask(detAll, sigma=sigma, mskThr=sigthr)
        mskAll = combMskImage(mskAll, detMskAll)

    """
    6. Remove the central object (or clear the central region)
       Separate the objects into different group and mask them out using
       different growth ratio

    central == 1: Only remove objects with central distance smaller than
                    a minimum value
    central == 2: Remove all objects within a very small radius
    """
    indCen = (np.where(cenDistComb < minCenDist) if central == 1
              else np.where(cenDistComb < galR1))
    sepFlags = addFlag(sepFlags, 'MULTICEN', len(indCen[0]) > 1)
    objNoCen = np.delete(objComb.copy(), indCen)
    r90NoCen = np.delete(r90.copy(), indCen)
    distNoCen = np.delete(cenDistComb.copy(), indCen)

    """
    7. Convert the list of SEP detections to initial guess of 1-Comp
        GALFIT model
    """
    """
    Group 1: Objects that are too close to the galaxy center
             Could be star or galaxy
    Group 2: Objects that are within certain radius, and flux is larger
             than certain fraction of the main galaxy (Near)
    Group 3: Objects that are within certain radius, and flux is larger
             than certain fraction of the main galaxy (Far)
    """
    fluxRatio1, fluxRatio2 = 0.10, 0.50
    group1 = np.where(cenDistComb <= galR50)
    group2 = np.where((cenDistComb > galR50) & (cenDistComb <= galR90) &
                      (objComb['cflux'] > fluxRatio1 * galFlux))
    group3 = np.where((cenDistComb > galR90) & (cenDistComb <= galR90 * 3.0) &
                      (objComb['cflux'] > fluxRatio2 * galFlux))
    sepFlags = addFlag(sepFlags, 'G1_EXIST', len(group1[0]) > 1)
    sepFlags = addFlag(sepFlags, 'G2_EXIST', len(group2[0]) > 0)
    sepFlags = addFlag(sepFlags, 'G3_EXIST', len(group3[0]) > 0)

    if verbose:
        print("###    N galaxies in Center / Group " +
              "1 / 2 / 3: %d%d/%d/%d" % (len(indCen[0]), len(group1[0]),
                                         len(group2[0]), len(group3[0])))

    # Number of galaxies should be fit altogether
    nObjFit = (len(group1[0]) + len(group2[0]) + len(group3[0]))
    iObjFit = np.concatenate((group1[0], group2[0], group3[0]))

    """
    8. Separate the rest objects into different groups according to
       their distance to the central galaxy

    Generating final mask by growing the size of objects using the
    correponding ratios:
        objG1 : growH
        objG2 : growW
        objG3 : growC
    """
    indG1 = (distNoCen <= galR1)
    indG2 = (distNoCen > galR1) & (distNoCen < galR3)
    indG3 = (distNoCen > galR3)
    objG1 = objNoCen[indG1]
    objG2 = objNoCen[indG2]
    objG3 = objNoCen[indG3]
    mskG1 = np.zeros(imgArr.shape, dtype='uint8')
    mskG2 = np.zeros(imgArr.shape, dtype='uint8')
    mskG3 = np.zeros(imgArr.shape, dtype='uint8')

    # Todo: Option Not Sure Which Way is Better!
    if growMethod == 1:
        sep.mask_ellipse(mskG1, objG1['x'], objG1['y'], objG1['a'], objG1['b'],
                         objG1['theta'], r=growH)
        sep.mask_ellipse(mskG2, objG2['x'], objG2['y'], objG2['a'], objG2['b'],
                         objG2['theta'], r=growW)
        sep.mask_ellipse(mskG3, objG3['x'], objG3['y'], objG3['a'], objG3['b'],
                         objG3['theta'], r=growC)
        objG1['a'] *= growH
        objG1['b'] *= growH
        objG2['a'] *= growW
        objG2['b'] *= growW
        objG3['a'] *= growC
        objG3['b'] *= growC
    else:
        adGrow1 = adaptiveMask(objG1, a=2.2)
        adGrow2 = adaptiveMask(objG2, a=2.4)
        adGrow3 = adaptiveMask(objG3, a=2.6)
        sep.mask_ellipse(mskG1, objG1['x'], objG1['y'], r90NoCen[indG1],
                         (r90NoCen[indG1] * objG1['b'] / objG1['a']),
                         objG1['theta'], r=adGrow1)
        sep.mask_ellipse(mskG2, objG2['x'], objG2['y'], r90NoCen[indG2],
                         (r90NoCen[indG2] * objG2['b'] / objG2['a']),
                         objG2['theta'], r=adGrow2)
        sep.mask_ellipse(mskG3, objG3['x'], objG3['y'], r90NoCen[indG3],
                         (r90NoCen[indG3] * objG3['b'] / objG3['a']),
                         objG3['theta'], r=adGrow3)
        objG1['a'] *= adGrow1
        objG1['b'] *= adGrow1
        objG2['a'] *= adGrow2
        objG2['b'] *= adGrow2
        objG3['a'] *= adGrow3
        objG3['b'] *= adGrow3

    # Combine the three groups of objects together
    if showAll:
        prefixF = os.path.join(rerunDir, (prefix + '_' + suffix + 'objFin'))
        objMask = np.concatenate((objG1, objG2, objG3))
        saveSEPObjects(objMask, prefix=prefixF, color='Green', csv=False,
                       pkl=False, reg=True)

    # Remove the segmentations of objects inside a radius:
    radLimit, magLimit = 20.0, 20.5

    # Hot one
    segHnew = copy.deepcopy(segH)
    objExcludeH = (np.where(cenDistH <= radLimit)[0] + 1)
    for index in objExcludeH:
        segHnew[segH == index] = 0

    # Remove the faint objects from the segmentation map
    for index, obj in enumerate(objH):
        if (-2.5 * np.log10(obj['cflux']) + photZP) >= magLimit:
            segHnew[segH == (index + 1)] = 0
    segMskH = seg2Mask(segHnew, sigma=(sigma + 1.0), mskThr=sigthr)

    # Cold One
    segCnew = copy.deepcopy(segC)
    objExcludeC = (np.where(cenDistC <= galR3)[0] + 1)
    for index in objExcludeC:
        segCnew[segC == index] = 0

    # Remove the faint objects from the segmentation map
    for index, obj in enumerate(objC):
        if (-2.5 * np.log10(obj['cflux']) + photZP) >= (magLimit - 1.5):
            segCnew[segC == (index + 1)] = 0
    segMskC = seg2Mask(segCnew, sigma=(sigma + 2.0), mskThr=sigthr)

    # Isolate the bright and/or big objects that are not too close to the
    # center
    segBig1 = copy.deepcopy(segC)
    for index, obj in enumerate(objC):
        if ((cenDistC[index] <= galR3) or (obj['flux'] <= galFlux * 0.2)):
            segBig1[segC == (index + 1)] = 0
    segMskBig1 = seg2Mask(segBig1, sigma=(sigma * 2.0 + 6.0), mskThr=sigthr)

    segBig2 = copy.deepcopy(segC)
    indBig2 = []
    for index, obj in enumerate(objC):
        if ((cenDistC[index] <= galR2) or (obj['flux'] <= galFlux * 0.3)):
            segBig2[segC == (index + 1)] = 0
        else:
            indBig2.append(index)
    segMskBig2 = seg2Mask(segBig2, sigma=(sigma + 1.0), mskThr=sigthr)
    objBig = objC[indBig2]

    """
    MultiMask Mode
    """
    if multiMask:
        # Larger mask
        objLG1 = objNoCen[indG1]
        objLG2 = objNoCen[indG2]
        objLG3 = objNoCen[indG3]
        mskLG1 = np.zeros(imgArr.shape, dtype='uint8')
        mskLG2 = np.zeros(imgArr.shape, dtype='uint8')
        mskLG3 = np.zeros(imgArr.shape, dtype='uint8')
        sep.mask_ellipse(
            mskLG1,
            objLG1['x'],
            objLG1['y'],
            objLG1['a'],
            objLG1['b'],
            objLG1['theta'],
            r=(growH + 0.2))
        sep.mask_ellipse(
            mskLG2,
            objLG2['x'],
            objLG2['y'],
            objLG2['a'],
            objLG2['b'],
            objLG2['theta'],
            r=(growW + 1.0))
        sep.mask_ellipse(
            mskLG3,
            objLG3['x'],
            objLG3['y'],
            objLG3['a'],
            objLG3['b'],
            objLG3['theta'],
            r=(growC + 1.5))
        segMskLH = seg2Mask(segHnew, sigma=(sigma + 1.5), mskThr=sigthr)
        segMskLC = seg2Mask(segCnew, sigma=(sigma + 2.5), mskThr=sigthr)
        segMskLB1 = seg2Mask(segBig1, sigma=(sigma * 2.0 + 8.0), mskThr=sigthr)
        segMskLB2 = seg2Mask(segBig2, sigma=(sigma + 2.0), mskThr=sigthr)
        if detFound:
            detLMsk = copy.deepcopy(detArr).astype(int)
            detLMsk[mskGal > 0] = 0
            detLMsk[detLMsk > 0] = 1
            detLMskConv = seg2Mask(detLMsk, sigma=4.5, mskThr=sigthr)
        objLG1['a'] *= (growH + 0.2)
        objLG1['b'] *= (growH + 0.2)
        objLG2['a'] *= (growW + 1.0)
        objLG2['b'] *= (growW + 1.0)
        objLG3['a'] *= (growC + 1.0)
        objLG3['b'] *= (growC + 1.0)

        # Small mask
        objSG1 = objNoCen[indG1]
        objSG2 = objNoCen[indG2]
        objSG3 = objNoCen[indG3]
        mskSG1 = np.zeros(imgArr.shape, dtype='uint8')
        mskSG2 = np.zeros(imgArr.shape, dtype='uint8')
        mskSG3 = np.zeros(imgArr.shape, dtype='uint8')
        sep.mask_ellipse(
            mskSG1,
            objSG1['x'],
            objSG1['y'],
            objSG1['a'],
            objSG1['b'],
            objSG1['theta'],
            r=(growH - 0.2))
        sep.mask_ellipse(
            mskSG2,
            objSG2['x'],
            objSG2['y'],
            objSG2['a'],
            objSG2['b'],
            objSG2['theta'],
            r=(growW - 1.0))
        sep.mask_ellipse(
            mskSG3,
            objSG3['x'],
            objSG3['y'],
            objSG3['a'],
            objSG3['b'],
            objSG3['theta'],
            r=(growC - 1.0))
        segMskSH = seg2Mask(segHnew, sigma=(sigma - 1.0), mskThr=sigthr)
        segMskSC = seg2Mask(segCnew, sigma=(sigma + 0.0), mskThr=sigthr)
        segMskSB1 = seg2Mask(segBig1, sigma=(sigma * 2.0), mskThr=sigthr)
        segMskSB2 = seg2Mask(segBig2, sigma=(sigma - 2.0), mskThr=sigthr)
        if detFound:
            detSMsk = copy.deepcopy(detArr).astype(int)
            detSMsk[mskGal > 0] = 0
            detSMsk[detSMsk > 0] = 1
            detSMskConv = seg2Mask(detSMsk, sigma=2.0, mskThr=sigthr)
        objSG1['a'] *= (growH - 0.2)
        objSG1['b'] *= (growH - 0.2)
        objSG2['a'] *= (growW - 1.0)
        objSG2['b'] *= (growW - 1.0)
        objSG3['a'] *= (growC - 1.0)
        objSG3['b'] *= (growC - 1.0)

    # Combine them into the final mask
    mskFinal = (mskG1 | mskG2 | mskG3 | segMskC | segMskH | segMskBig1 |
                segMskBig2)
    mskAll = (mskAll | mskG1 | mskG2 | mskG3 | segMskC | segMskH |
              segMskBig1 | segMskBig2)

    if multiMask:
        mskSmall = (mskSG1 | mskSG2 | mskSG3 | segMskSC | segMskSH |
                    segMskSB1 | segMskSB2)
        mskLarge = (mskLG1 | mskLG2 | mskLG3 | segMskLC | segMskLH |
                    segMskLB1 | segMskLB2)

    # Bright star mask:
    ellStar = None
    if starCat is not None:
        xStar, yStar, rStar = matchStarCatalog(starCat, imgHead)
        if (xStar is not None) and (yStar is not None):
            print("###    There are %d bright stars on image" % len(xStar))
            mskFinal = maskStars(mskFinal.astype('uint8'),
                                 xStar, yStar, rStar)
            # Be more aggressive for the all objects mask
            mskAll = maskStars(mskAll.astype('uint8'),
                               xStar, yStar, rStar * 1.2)
            if multiMask:
                mskSmall = maskStars(mskSmall.astype('uint8'),
                                     xStar, yStar, rStar)
                mskLarge = maskStars(mskLarge.astype('uint8'),
                                     xStar, yStar, rStar)
            ellStar = [
                Ellipse(
                    xy=np.array([xStar[i], yStar[i]]),
                    width=np.array(2.0 * rStar[i]),
                    height=np.array(2.0 * rStar[i]),
                    angle=np.array(0.0))
                for i in range(len(xStar))
            ]

    # if extMask is provided, combine them
    if extMask is not None:
        mskFinal = (mskFinal | extMask)
        mskAll = (mskAll | extMask)
        if multiMask:
            mskSmall = (mskSmall | extMask)
            mskLarge = (mskLarge | extMask)

    # Have the option to combine with HSC BAD MASK
    if combBad and badFound:
        mskFinal = (mskFinal | mskArr)
        if multiMask:
            mskSmall = (mskSmall | mskArr)
            mskLarge = (mskLarge | mskArr)

    # Have the option to combine with HSC DETECTION MASK
    if combDet and detFound:
        mskFinal = (mskFinal | detMsk)
        if multiMask:
            mskSmall = (mskSmall | detSMskConv)
            mskLarge = (mskLarge | detLMskConv)

    # if extKeep is provided, free them
    if extKeep is not None:
        mskFinal[extKeep > 0] = 0
        if multiMask:
            mskSmall[extKeep > 0] = 0
            mskLarge[extKeep > 0] = 0

    # Mask out all the NaN pixels
    mskFinal[indImgNaN] = 1
    mskFinFile = os.path.join(rerunDir,
                              (prefix + '_' + suffix + 'mskfin.fits'))
    if multiMask:
        mskSmall[indImgNaN] = 1
        mskLarge[indImgNaN] = 1
        mskSmallFile = mskFinFile.replace('mskfin', 'msksmall')
        mskLargeFile = mskFinFile.replace('mskfin', 'msklarge')

    # See if the center of the image has been masked out
    sumMskCen, dump1, dump2 = sep.sum_ellipse(
        np.float32(mskFinal),
        galCenX,
        galCenY,
        20.0,
        20.0, (galPA * np.pi / 180.0),
        r=1.0)
    sepFlags = addFlag(sepFlags, 'MSK_CEN', sumMskCen > 0)

    sumMskR20, dump1, dump2 = sep.sum_ellipse(
        np.float32(mskFinal),
        galCenX,
        galCenY,
        galR20, (galR20 * galQ), (galPA * np.pi / 180.0),
        r=1.0)
    sepFlags = addFlag(sepFlags, 'MSK_R20', sumMskR20 > 0)

    sumMskR50, dump1, dump2 = sep.sum_ellipse(
        np.float32(mskFinal),
        galCenX,
        galCenY,
        galR50, (galR50 * galQ), (galPA * np.pi / 180.0),
        r=1.0)
    sepFlags = addFlag(sepFlags, 'MSK_R50', sumMskR50 > 0)
    if verbose:
        print("###    N pixels are masked out " +
              "<20pix / <R20 / <R50: %d / %d / %d" % (sumMskCen,
                                                      sumMskR20, sumMskR50))

    # Add a few information about the central galaxy to the header
    mskHead = copy.deepcopy(imgHead)
    mskHead.set('GAL_X', galX)
    mskHead.set('GAL_Y', galY)
    mskHead.set('GAL_CENX', galCenX)
    mskHead.set('GAL_CENY', galCenY)
    mskHead.set('GAL_FLUX', galFlux)
    mskHead.set('GAL_Q', galQ)
    mskHead.set('GAL_A', galA)
    mskHead.set('GAL_PA', galPA)
    mskHead.set('GAL_R20', galR20)
    mskHead.set('GAL_R50', galR50)
    mskHead.set('GAL_R90', galR90)
    mskHead.set('GAL_R1', galR1)
    mskHead.set('GAL_R2', galR2)
    mskHead.set('GAL_R3', galR3)
    mskHead.set('NUM_FIT', nObjFit)

    # Put the Flags into the header
    for flag in sepFlags:
        mskHead.set(flag['name'], flag['value'])

    # Save masks to FITS
    mskAllFile = os.path.join(rerunDir,
                              (prefix + '_' + suffix + 'mskall.fits'))
    mskAll = mskAll.astype('uint8')
    saveFits(mskAll, mskAllFile, head=imgHead)

    mskFinal = mskFinal.astype('uint8')
    saveFits(mskFinal, mskFinFile, head=mskHead)
    if multiMask:
        mskSmall = mskSmall.astype('uint8')
        saveFits(mskSmall, mskSmallFile, head=mskHead)
        mskLarge = mskLarge.astype('uint8')
        saveFits(mskLarge, mskLargeFile, head=mskHead)

    if visual:
        if showAll:
            # Cold Background
            showSEPImage(bkgC.back(), contrast=0.3,
                         title='Background - Cold Run',
                         pngName=os.path.join(rerunDir,
                                              (prefix + '_' +
                                               suffix + 'bkgC.png')))
            # Hot Background
            showSEPImage(bkgH.back(), contrast=0.3,
                         title='Background - Hot Run',
                         pngName=os.path.join(rerunDir,
                                              (prefix + '_' +
                                               suffix + 'bkgH.png')))

            objPNG1 = os.path.join(rerunDir,
                                   (prefix + '_' + suffix + 'objC.png'))
            objPNG2 = os.path.join(rerunDir,
                                   (prefix + '_' + suffix + 'objH.png'))
            objEllC = getEll2Plot(objC, radius=(objC['a'] / 2 * growC))
            objEllH = getEll2Plot(objH, radius=(objH['a'] / 2 * growH))
            # Cold Detections
            showSEPImage(imgSubC, contrast=0.15, title='Detections - Cold Run',
                         pngName=objPNG1, ellList1=objEllC, ellColor1='b')

            # Hot Detections
            showSEPImage(imgSubH, contrast=0.30, title='Detections - Hot Run',
                         pngName=objPNG2, ellList1=objEllH, ellColor1='r')

            # Combined Detections
            objPNG3 = os.path.join(rerunDir,
                                   (prefix + '_' + suffix + 'objComb.png'))
            objEllComb = getEll2Plot(objComb,
                                     radius=(objComb['a'] / 2 * growH))
            showSEPImage(imgSubC, contrast=0.1, title='Detections - Combined',
                         pngName=objPNG3, ellList1=objEllComb,
                         ellColor1='orange')

            # R20/50/90 of each object
            objPNG4 = os.path.join(rerunDir,
                                   (prefix + '_' + suffix + 'objRad.png'))
            objEllR20 = getEll2Plot(objComb, radius=r50)
            objEllR50 = getEll2Plot(objComb, radius=r90)
            objEllR90 = getEll2Plot(objComb[overSize], radius=r90[overSize])
            # Add three ellipses to highlight galR1, R2, & R3
            ell1 = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR1 * galQ),
                height=(2.0 * galR1),
                angle=(galPA + 90.0))
            ell2 = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR2 * galQ),
                height=(2.0 * galR2),
                angle=(galPA + 90.0))
            ell3 = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR3 * galQ),
                height=(2.0 * galR3),
                angle=(galPA + 90.0))
            showSEPImage(
                imgArr,
                contrast=0.20,
                title='Flux Radius: R20/R50/R90',
                pngName=objPNG4,
                ellList1=objEllR20,
                ellColor1='r',
                ellList2=objEllR50,
                ellColor2='orange',
                ellList3=objEllR90,
                ellColor3='b',
                ell1=ell1,
                ell2=ell2,
                ell3=ell3)

        # Mask of all objects
        mskPNG1 = os.path.join(rerunDir,
                               (prefix + '_' + suffix + 'mskall.png'))
        showSEPImage(imgSubC, contrast=0.75, title='Mask - All Objects',
                     pngName=mskPNG1, mask=mskAll)

        # Final mask
        mskPNG2 = os.path.join(rerunDir,
                               (prefix + '_' + suffix + 'mskfin.png'))
        ellA = Ellipse(xy=(galX, galY), width=(2.0 * galR1 * galQ),
                       height=(2.0 * galR1), angle=(galPA + 90.0))
        ellB = Ellipse(xy=(galX, galY), width=(2.0 * galR2 * galQ),
                       height=(2.0 * galR2), angle=(galPA + 90.0))
        ellC = Ellipse(xy=(galX, galY), width=(2.0 * galR3 * galQ),
                       height=(2.0 * galR3), angle=(galPA + 90.0))

        # objEllG1 = getEll2Plot(objG1)
        objEllG2 = getEll2Plot(objG2)
        objEllG3 = getEll2Plot(objG3)
        objEllBig = getEll2Plot(objBig)

        showSEPImage(
            imgArr,
            contrast=0.75,
            title='Mask - Final',
            pngName=mskPNG2,
            mask=mskFinal,
            ellList1=objEllBig,
            ellColor1='green',
            ellList2=objEllG2,
            ellColor2='orange',
            ellList3=objEllG3,
            ellColor3='m',
            ellList4=ellStar,
            ell1=ellA,
            ell2=ellB,
            ell3=ellC,
            ellColor4='b')

        # Statistics of the detected objects
        objPNG = os.path.join(rerunDir, (prefix + '_' + suffix + 'objs.png'))
        showObjects(objComb, cenDistComb, rad=r90, outPNG=objPNG,
                    cenInd=cenObjIndex, r1=galR50, r2=galR90,
                    r3=(3.0 * galR90),
                    fluxRatio1=fluxRatio1, fluxRatio2=fluxRatio2,
                    prefix=prefix, highlight=iObjFit)

        if multiMask:
            mskPNG3 = mskPNG2.replace('mskfin', 'msksmall')
            mskPNG4 = mskPNG2.replace('mskfin', 'msklarge')

            ellA = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR1 * galQ),
                height=(2.0 * galR1),
                angle=(galPA + 90.0))
            ellB = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR2 * galQ),
                height=(2.0 * galR2),
                angle=(galPA + 90.0))
            ellC = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR3 * galQ),
                height=(2.0 * galR3),
                angle=(galPA + 90.0))
            objEllSG1 = getEll2Plot(objSG1)
            objEllSG2 = getEll2Plot(objSG2)
            objEllSG3 = getEll2Plot(objSG3)

            showSEPImage(
                imgArr,
                contrast=0.75,
                title='Mask - Small',
                pngName=mskPNG3,
                mask=mskSmall,
                ellList1=objEllSG1,
                ellColor1='green',
                ellList2=objEllSG2,
                ellColor2='orange',
                ellList3=objEllSG3,
                ellColor3='m',
                ell1=ellA,
                ell2=ellB,
                ell3=ellC,
                ellColor4='b')

            ellA = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR1 * galQ),
                height=(2.0 * galR1),
                angle=(galPA + 90.0))
            ellB = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR2 * galQ),
                height=(2.0 * galR2),
                angle=(galPA + 90.0))
            ellC = Ellipse(
                xy=(galX, galY),
                width=(2.0 * galR3 * galQ),
                height=(2.0 * galR3),
                angle=(galPA + 90.0))
            objEllLG1 = getEll2Plot(objLG1)
            objEllLG2 = getEll2Plot(objLG2)
            objEllLG3 = getEll2Plot(objLG3)

            showSEPImage(
                imgArr,
                contrast=0.75,
                title='Mask - Large',
                pngName=mskPNG4,
                mask=mskLarge,
                ellList1=objEllLG1,
                ellColor1='green',
                ellList2=objEllLG2,
                ellColor2='orange',
                ellList3=objEllLG3,
                ellColor3='m',
                ell1=ellA,
                ell2=ellB,
                ell3=ellC,
                ellColor4='b')


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
        '-rerun',
        '--rerun',
        dest='rerun',
        help='Name of the rerun',
        default='default')
    parser.add_argument(
        '-k',
        dest='kernel',
        help='SExtractor detection kernel',
        type=int,
        default=6,
        choices=range(1, 8))
    parser.add_argument(
        '-c',
        dest='central',
        help='Method to clean the central region',
        type=int,
        default=1,
        choices=range(1, 3))
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
        choices=range(1, 2))
    parser.add_argument(
        '--bkgH',
        dest='bSizeH',
        help='Background size for the Hot Run',
        type=int,
        default=20)
    parser.add_argument(
        '--bkgC',
        dest='bSizeC',
        help='Background size for the Cold Run',
        type=int,
        default=50)
    parser.add_argument(
        '--thrH',
        dest='thrH',
        help='Detection threshold for the Hot Run',
        type=float,
        default=2.0)
    parser.add_argument(
        '--thrC',
        dest='thrC',
        help='Detection threshold for the Cold Run',
        type=float,
        default=5.0)
    parser.add_argument(
        '--growC',
        dest='growC',
        help='Ratio of Growth for the Cold Objects',
        type=float,
        default=5.0)
    parser.add_argument(
        '--growW',
        dest='growW',
        help='Ratio of Growth for the Warm Objects',
        type=float,
        default=3.0)
    parser.add_argument(
        '--growH',
        dest='growH',
        help='Ratio of Growth for the Hot Objects',
        type=float,
        default=1.5)
    parser.add_argument(
        '--minDetC',
        dest='minDetC',
        help='Minimum pixels for Cold Detections',
        type=float,
        default=9.0)
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
        default=64.0)
    parser.add_argument(
        '--debThrH',
        dest='debThrH',
        help='Deblending threshold for the Hot Run',
        type=float,
        default=64.0)
    parser.add_argument(
        '--debConC',
        dest='debConC',
        help='Deblending continuum level for the Cold Run',
        type=float,
        default=0.003)
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
        default=5.0)
    parser.add_argument(
        '--galR2',
        dest='galR2',
        help='galR2 = galR2 * galR90',
        type=float,
        default=10.0)
    parser.add_argument(
        '--galR3',
        dest='galR3',
        help='galR3 = galR3 * galR90',
        type=float,
        default=16.0)
    parser.add_argument(
        '--sigma',
        dest='sigma',
        help='Sigma to Gaussian smooth the segmentation image',
        type=float,
        default=5.0)
    parser.add_argument(
        '--sigthr',
        dest='sigthr',
        help='Threshold for constraing the convolution mask',
        type=float,
        default=0.01)
    parser.add_argument(
        '--multiMask', dest='multiMask', action="store_true", default=False)
    parser.add_argument(
        '--noBkgC', dest='noBkgC', action="store_true", default=False)
    parser.add_argument(
        '--noBkgH', dest='noBkgH', action="store_true", default=False)
    parser.add_argument(
        '--useSigArr', dest='useSigArr', action="store_true", default=False)
    parser.add_argument(
        '--combBad', dest='combBad', action="store_true", default=True)
    parser.add_argument(
        '--combDet', dest='combDet', action="store_true", default=True)
    parser.add_argument(
        '--showAll', dest='showAll', action="store_true", default=False)
    parser.add_argument(
        '--brightStar', dest='brightStar', default=None,
        help='Bright star catalog')
    parser.add_argument(
        '--regMask',
        dest='regMask',
        default=None,
        help='DS9 regions to be masked')
    parser.add_argument(
        '--regKeep',
        dest='regKeep',
        default=None,
        help='DS9 regions to be kept')

    args = parser.parse_args()

    coaddCutoutPrepare(
        args.prefix,
        root=args.root,
        bSizeH=args.bSizeH,
        bSizeC=args.bSizeC,
        thrH=args.thrH,
        thrC=args.thrC,
        growMethod=args.grow,
        growH=args.growH,
        growW=args.growW,
        growC=args.growC,
        kernel=args.kernel,
        central=args.central,
        maskMethod=args.mask,
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
        rerun=args.rerun,
        sigma=args.sigma,
        sigthr=args.sigthr,
        galR1=args.galR1,
        galR2=args.galR2,
        galR3=args.galR3,
        regMask=args.regMask,
        regKeep=args.regKeep,
        showAll=args.showAll,
        brightStar=args.brightStar,
        multiMask=args.multiMask)
