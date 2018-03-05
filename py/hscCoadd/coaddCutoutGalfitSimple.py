#!/usr/bin/env python
# encoding: utf-8
"""Fit simple 2-D models to HSC cutouts."""

from __future__ import division

import os
import copy
import shutil
import argparse
import warnings
import subprocess
import numpy as np
from distutils import spawn

# Matplotlib related
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Ellipse

# Astropy
from astropy.io import fits

# Personal
import hscUtils as hUtil
import galfitParser as gPar

# Colors and color maps
from palettable.colorbrewer.qualitative import Set1_9 as compColor
try:
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('k', 1.)
    cmap2 = plt.get_cmap('inferno')
    cmap2.set_bad('w', 1.)
except Exception:
    from palettable.cubehelix import Cubehelix
    cmap = Cubehelix.make(start=0.3, rotation=-0.5,
                          reverse=True).mpl_colormap
    cmap.set_bad('k', 1.)
    cmap2 = Cubehelix.make(start=0.5, rotation=-1.5,
                           gamma=1.0, sat=1.2,
                           min_light=0.,
                           max_light=1.).mpl_colormap
    cmap2.set_bad('w', 1.)

mpl.rcParams['figure.figsize'] = 12, 10
mpl.rcParams['xtick.major.size'] = 8.0
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.size'] = 4.0
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 8.0
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.size'] = 4.0
mpl.rcParams['ytick.minor.width'] = 1.5
plt.rc('text', usetex=True)
mpl.rc('axes', linewidth=2)
plt.ioff()


COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def galfitAIC(galOut):
    u"""
    Estimate the AIC, BIC, and HQ for a GALFIT model.

    AIC=-2 ln(L) + 2 k : akaike information criterion
    BIC=-2 ln(L) + ln(n)*k : bayesian information criterion
    HQ=-2 ln(L) + ln(ln(n))*k  hannan-quinn criterion
    """
    chisq = np.float(galOut.chisq)
    ndof = np.float(galOut.ndof)
    nfree = np.float(galOut.nfree)
    # AIC
    aic = chisq + 2.0 * nfree + (2.0 * nfree * (nfree + 1.0))/(ndof -
                                                               nfree - 1.0)
    # BIC
    bic = chisq + np.log(ndof) * nfree
    # Hannan-Quinn cirterion
    hq = chisq + np.log(np.log(ndof)) * nfree

    return aic, bic, hq


def showModels(outFile, galOut, root=None, verbose=True, vertical=False,
               showZoom=True, zoomLimit=6.0, showTitle=True, showChi2=True,
               zoomSize=None, overComp=True, maskRes=True, returnPlot=False,
               scale1=0.03, sample1=500,
               scale2=0.40, sample2=500, title=None):
    """
    Three columns view of the Galfit models.

    With overlapped ellipse for each component
    """
    if (root is not None) and (os.path.dirname(outFile) == ''):
        outFile = os.path.join(root, outFile)
    elif root is None:
        root = os.path.dirname(outFile)
    else:
        root = ''
    """  Read in the results fits file """
    arrOut = fits.open(outFile)
    """ Name of the output PNG figure """
    outPNG = outFile.replace('.fits', '.png')

    if verbose:
        print " ## %s ---> %s " % (outFile, outPNG)

    """ Layout of the figure """
    if not vertical:
        reg1 = [0.05, 0.05, 0.30, 0.90]
        reg2 = [0.35, 0.05, 0.30, 0.90]
        reg3 = [0.65, 0.05, 0.30, 0.90]
        xsize, ysize = 18, 6
    else:
        reg1 = [0.05, 0.65, 0.90, 0.30]
        reg2 = [0.05, 0.35, 0.90, 0.30]
        reg3 = [0.05, 0.05, 0.90, 0.30]
        xsize, ysize = 6, 18

    """ Set up the figure """
    fig = plt.figure(figsize=(xsize, ysize))
    ax1 = fig.add_axes(reg1)
    ax2 = fig.add_axes(reg2)
    ax3 = fig.add_axes(reg3)

    """ Geometry of each component """
    nComp = galOut.num_components
    compX, compY, compR, compQ, compPA = [], [], [], [], []
    for i in range(1, nComp+1):
        compStr = 'component_' + str(i)
        compInfo = getattr(galOut, compStr)
        if compInfo.component_type == 'sersic':
            compX.append(compInfo.xc)
            compY.append(compInfo.yc)
            compR.append(compInfo.re)
            compQ.append(compInfo.ar)
            compPA.append(compInfo.pa)

    """ Image size and scale """
    imgOri = arrOut[1].data
    imgMod = arrOut[2].data
    imgRes = arrOut[3].data
    imgX, imgY = imgOri.shape

    if maskRes:
        maskFile = os.path.join(root, galOut.input_mask)
        if (os.path.isfile(maskFile)) or os.path.islink(maskFile):
            mskArr = fits.open(maskFile)[0].data
            imgMsk = mskArr[np.int(galOut.box_x0)-1:np.int(galOut.box_x1),
                            np.int(galOut.box_y0)-1:np.int(galOut.box_y1)]
            resShow = copy.deepcopy(imgRes)
            resShow[imgMsk > 0] = np.nan
        else:
            print "XXX Can not find the mask file : %s" % maskFile
            mskArr = None
            resShow = imgRes
    else:
        resShow = imgRes
        mskArr = None

    maxR = (np.max(np.asarray(compR)) * zoomLimit)
    if zoomSize is not None:
        zoomR = zoomSize / 2.0
        imgOri = imgOri[np.int(imgX/2.0 - zoomR):np.int(imgY/2.0 + zoomR),
                        np.int(imgY/2.0 - zoomR):np.int(imgY/2.0 + zoomR)]
        imgMod = imgMod[np.int(imgX/2.0 - zoomR):np.int(imgY/2.0 + zoomR),
                        np.int(imgY/2.0 - zoomR):np.int(imgY/2.0 + zoomR)]
        resShow = resShow[np.int(imgX/2.0 - zoomR):np.int(imgY/2.0 + zoomR),
                          np.int(imgY/2.0 - zoomR):np.int(imgY/2.0 + zoomR)]
        xPad = np.int(imgX/2.0 - zoomR)
        yPad = np.int(imgY/2.0 - zoomR)
    elif (imgX/2.0 >= maxR) and (imgY/2.0 >= maxR) and showZoom:
        imgOri = imgOri[np.int(imgX/2.0 - maxR):np.int(imgY/2.0 + maxR),
                        np.int(imgY/2.0 - maxR):np.int(imgY/2.0 + maxR)]
        imgMod = imgMod[np.int(imgX/2.0 - maxR):np.int(imgY/2.0 + maxR),
                        np.int(imgY/2.0 - maxR):np.int(imgY/2.0 + maxR)]
        resShow = resShow[np.int(imgX/2.0 - maxR):np.int(imgY/2.0 + maxR),
                          np.int(imgY/2.0 - maxR):np.int(imgY/2.0 + maxR)]
        xPad = np.int(imgX/2.0 - maxR)
        yPad = np.int(imgY/2.0 - maxR)
        print " ## Image has been truncated to highlight the galaxy !"
    else:
        xPad, yPad = 0, 0

    imin1, imax1 = hUtil.zscale(imgOri, contrast=scale1,
                                samples=sample1)
    imin2, imax2 = hUtil.zscale(imgMod, contrast=(scale1 + 0.02),
                                samples=sample1)
    imin3, imax3 = hUtil.zscale(resShow, contrast=scale2,
                                samples=sample2)

    compX = np.asarray(compX) - np.float(galOut.box_x0)
    compY = np.asarray(compY) - np.float(galOut.box_y0)
    compR = np.asarray(compR)
    compQ = np.asarray(compQ)
    compPA = np.asarray(compPA)
    compX -= xPad
    compY -= yPad

    """ 1. Origin Image """
    ax1.xaxis.set_major_formatter(NullFormatter())
    ax1.yaxis.set_major_formatter(NullFormatter())
    ax1.imshow(np.arcsinh(imgOri), interpolation="none",
               vmax=imax1, cmap=cmap, vmin=imin1, origin='lower')

    if overComp:
        try:
            for ii in range(len(compX)):
                x0, y0, r0 = compX[ii], compY[ii], compR[ii]
                q0, pa0 = compQ[ii], compPA[ii]
                ellRe = Ellipse(xy=(x0, y0), width=(r0*q0*2.0),
                                height=(r0*2.0), angle=pa0)
                ax1.add_artist(ellRe)
                ellRe.set_clip_box(ax1.bbox)
                ellRe.set_alpha(1.0)
                ellRe.set_edgecolor(compColor.mpl_colors[ii])
                ellRe.set_facecolor('none')
                ellRe.set_linewidth(2.5)
        except Exception:
            print "XXX Can not highlight the components"

    if showTitle:
        if title is None:
            title = os.path.basename(outFile)
        titleStr = ax1.text(0.50, 0.90,
                            r'$\mathrm{%s}$' % title,
                            fontsize=25, transform=ax1.transAxes,
                            horizontalalignment='center')
        titleStr.set_bbox(dict(facecolor='white', alpha=0.6,
                          edgecolor='white'))

    """ 2. Model Image """
    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2.yaxis.set_major_formatter(NullFormatter())
    if imax1 > imin2:
        ax2.imshow(np.arcsinh(imgMod), interpolation="none",
                   vmax=imax1, cmap=cmap, vmin=imin2, origin='lower')
    else:
        ax2.imshow(np.arcsinh(imgMod), interpolation="none",
                   vmax=imax2, cmap=cmap, vmin=imin2, origin='lower')
    """ Contour """
    try:
        tam = np.size(imgMod, axis=0)
        contour_x = np.arange(tam)
        contour_y = np.arange(tam)
        ax2.contour(contour_x, contour_y, np.arcsinh(imgMod), colors='c',
                    linewidths=1.5)
    except Exception:
        print "XXX Can not generate the Contour !"

    """ Show the reduced chisq """
    if showChi2:
        ax2.text(0.06, 0.92, '${\chi}^2/N_{DoF}$ : %s' % galOut.reduced_chisq,
                 fontsize=14, transform=ax2.transAxes)
        aic, bic, hq = galfitAIC(galOut)
        if verbose:
            print "  # AIC : ", aic
            print "  # BIC : ", bic
            print "  # HQ : ", hq
        ax2.text(0.06, 0.87, 'AIC : %9.3f' % aic,
                 fontsize=14, transform=ax2.transAxes)
        ax2.text(0.06, 0.82, 'BIC : %9.3f' % bic,
                 fontsize=14, transform=ax2.transAxes)
        ax2.text(0.06, 0.77, 'HQ : %9.3f' % hq,
                 fontsize=14, transform=ax2.transAxes)

    """ 3. Residual Image """
    ax3.xaxis.set_major_formatter(NullFormatter())
    ax3.yaxis.set_major_formatter(NullFormatter())
    ax3.imshow(np.arcsinh(resShow), interpolation="none",
               vmin=imin3, vmax=imax3, cmap=cmap,
               origin='lower')
    try:
        for ii in range(len(compX)):
            x0, y0 = compX[ii], compY[ii]
            r0, q0, pa0 = compR[ii], compQ[ii], compPA[ii]
            ellRe = Ellipse(xy=(x0, y0), width=(r0*q0*2.0), height=(r0*2.0),
                            angle=pa0)
            ax3.add_artist(ellRe)
            ellRe.set_clip_box(ax3.bbox)
            ellRe.set_alpha(1.0)
            ellRe.set_edgecolor(compColor.mpl_colors[ii])
            ellRe.set_facecolor('none')
            ellRe.set_linewidth(2.5)
    except Exception:
        print "XXX Can not highlight the components"

    """ Save Figure """
    fig.savefig(outPNG, dpi=80)

    plt.close(fig)

    """ Clean the large image file """
    del imgOri, imgMod, imgRes, resShow
    del mskArr

    return fig


def removePSF(readin, root=None, verbose=True, abspath=False, run=False):
    """
    Remove the PSF convolution from the GALFIT read in file.

    Parameters:
    """
    if abspath:
        readin = readin
    elif root is not None:
        readin = os.path.join(root, readin)

    if '.in' in readin:
        readinNew = readin.replace('.in', '_nopsf.in')
    else:
        readinNew = readin + '_nopsf'
    if verbose:
        print ' ## %s ---> %s' % (readin, readinNew)

    inFile = open(readin, 'r')
    temp = inFile.readlines()

    for i in range(len(temp)):
        if 'D)' in temp[i]:
            temp[i] = 'D)  # Input PSF image \n'

    outFile = open(readinNew, 'w')
    for line in temp:
        outFile.write(line)
    outFile.close()

    return readinNew


def log2Readin(outFile, logFile, iniFile, root=None,
               verbose=True, abspath=False):
    """
    Back up the original input file.

    Update the readin file using the output log file
    """
    if root is not None:
        logFile = os.path.join(root, logFile)
    if os.path.isfile(logFile):
        if abspath:
            iniFile = iniFile
        elif root is not None:
            iniFile = os.path.join(root,
                                   os.path.basename(iniFile))
        else:
            iniFile = iniFile

        if os.path.basename(iniFile) in open(logFile).read():
            if verbose:
                print ' ## %s  ---> %s ' % (logFile, iniFile)
            shutil.copyfile(iniFile, iniFile + '_back')
            shutil.copyfile(logFile, iniFile)
        else:
            warnings.warn('XXX Maybe the wrong log file? Please check! %s' %
                          logFile)
    else:
        warnings.warn('XXX Can not find the log file: %s' % logFile)

    return iniFile


def generateSubcomp(readFile, root=None, galfit=None, separate=True,
                    verbose=True, abspath=False):
    """Run Galfit -o3 to generate the model image of each component."""
    """ Find GALFIT """
    if galfit is None:
        galfit = spawn.find_executable('galfit')
        if galfit is None:
            print WAR
            raise Exception("XXX Can not find the GALFIT executable")

    """ Check the Read-in File """
    if not os.path.isfile(readFile):
        print WAR
        raise Exception("XXX Can not find the READIN file: %s", readFile)
    """ Absolute path of the read in file """
    if abspath:
        readFile = os.path.abspath(readFile)

    """ GALFIT command """
    galfitCommand = galfit + ' -o3 ' + readFile

    """ Excecute the command """
    if verbose:
        print "### Generating subcomp.fits file ..."
    if root is not None:
        proc = subprocess.Popen([galfitCommand], cwd=root, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen([galfitCommand], shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    proc.communicate()
    print "###  ... DONE !"

    """ Find the subcomp.fits"""
    if root is None:
        root = os.path.dirname(readFile)
    subFile = os.path.join(root, 'subcomps.fits')
    if not os.path.isfile(subFile):
        print WAR
        raise Exception('XXX Can not find the SUBCOMP file: %s', subFile)

    """Separate the subcomps.fits into fits image of individual component"""
    if separate:
        prefix = os.path.basename(readFile)
        prefix = os.path.splitext(prefix)[0]
        prefix = os.path.join(root, prefix)
        subComp = fits.open(subFile)
        subHeader = subComp[1].header
        nComp = len(subComp)
        for i in range(1, nComp):
            print SEP
            print " ## Component : %d " % i
            compFits = prefix + '_comp' + str(i) + '.fits'
            print "  #    Saved to %s " % compFits
            compArr = subComp[i].data
            compHdu = fits.PrimaryHDU(compArr, header=subHeader)
            hduList = fits.HDUList([compHdu])
            hduList.writeto(compFits, overwrite=True)

    """ Remove the subcomp.fits file """
    os.remove(subFile)

    return


def getCenConstrFile(comps, location=None, name=None):
    """Generate simple central constraints file."""
    compArr = []
    compStr = ''
    for comp in comps:
        compArr.append(np.int16(comp))
        compStr += (comp + '_')
    maxN = str(np.max(compArr))
    compStr = compStr[0:-1]

    if name is None:
        name = maxN + 'comp.cons'

    if location is not None:
        name = os.path.join(location, name)

    x_str = compStr + '   x      offset     # Hard constraint'
    y_str = compStr + '   y      offset     # Hard constraint'
    constrStr = [x_str, y_str]

    f = open(name, 'w')

    f.write('\n')
    f.write('# Component/    parameter   constraint	Comment \n')
    f.write('# operation	(see below)   range \n')
    f.write(x_str + '\n')
    f.write(y_str + '\n')
    f.write('\n')

    f.close()

    return constrStr


def coaddRunGalfit(readFile, root=None, imax=150, galfit=None, updateRead=True,
                   keepLog=True, show=True, expect=None, showZoom=False,
                   zoomSize=None, removePsf=True, verbose=False,
                   abspath=False, deleteAfter=False, savePkl=True,
                   scale1=0.03, scale2=0.40):
    """Run GALFIT."""
    """ Find GALFIT """
    if galfit is None:
        galfit = spawn.find_executable('galfit')
        if galfit is None:
            print WAR
            raise Exception("XXX Can not find the GALFIT executable")

    """ Check the Read-in File """
    if not os.path.isfile(readFile):
        raise Exception("XXX Can not find the READIN file: %s", readFile)
    """ Absolute path of the read in file """
    if abspath:
        readFile = os.path.abspath(readFile)

    """ Expected output model """
    if expect is None:
        expect = readFile.replace('.in', '.fits')
    if os.path.isfile(expect):
        os.remove(expect)

    """ IMAX string """
    imaxStr = " -imax %4d" % imax

    """ GALFIT command """
    galfitCommand = galfit + ' ' + imaxStr + ' ' + readFile
    if verbose:
        print " ## Command : %s" % galfitCommand

    """ Excecute the command """
    if (root is not None) and abspath:
        proc = subprocess.Popen([galfitCommand], cwd=root, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen([galfitCommand], shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    if keepLog:
        logFile = readFile + '.log'
        f = open(logFile, 'w')
        for line in proc.stdout.readlines():
            f.write(line)
        f.close()
    else:
        for line in proc.stdout.readlines():
            print line
        proc.wait()

    if not os.path.isfile(expect):
        done = False
        print COM
        print "### GALFIT run failed for : %s" % readFile
        print COM
    else:
        done = True
        print COM
        print "### GALFIT run finished : %s" % expect
        print COM

        """  Read in the output file """
        galOut = gPar.GalfitResults(expect)
        galLog = galOut.logfile
        iniFile = galOut.input_initfile

        """ Update the read in file """
        if updateRead:
            newReadIn = log2Readin(expect, galLog, iniFile,
                                   root=None, verbose=True,
                                   abspath=abspath)
            if verbose:
                print "## Read-in file has been updated : %s" % newReadIn
            """ Remove the PSF from the new readin file """
            if removePSF:
                noPsfReadIn = removePSF(readFile, root=None, verbose=True,
                                        abspath=abspath)
                if verbose:
                    print "## NoPSF read-in is available : %s" % noPsfReadIn

        """ MORE CAN BE DONE HERE """

        """ Save a Pickle version of the output parameters """
        if savePkl:
            outPkl = expect.replace('.fits', '.pkl')
            hUtil.saveToPickle(expect, outPkl)

        """ Visualization of the model """
        if show:
            try:
                showModels(expect, galOut, root=root, verbose=True,
                           vertical=False, showZoom=showZoom, showTitle=True,
                           showChi2=True, overComp=True, maskRes=True,
                           zoomSize=zoomSize,
                           scale1=scale1, scale2=scale2)
                plotOk = True
            except Exception:
                plotOk = False
                print "XXX No plot is made !"
        else:
            plotOk = False

        """ Delete After """
        if deleteAfter:
            os.remove(expect)

    return done, plotOk


def imgSameSize(img1, img2):
    """Check if two images have the same size."""
    dimX1, dimY1 = img1.shape
    dimX2, dimY2 = img2.shape
    if (dimX1 == dimX2) and (dimY1 == dimY2):
        return True
    else:
        return False


def readSbpInput(prefix, root=None, rerun='default',
                 maskType='mskfin', extMsk=None, imgSub=False):
    """Parse input data."""
    # Get the names of necessary input images
    if imgSub:
        imgFile = prefix + '_imgsub.fits'
    else:
        imgFile = prefix + '_img.fits'

    if extMsk is not None:
        mskFile = extMsk
    else:
        mskFile = prefix + '_' + maskType + '.fits'
        if root is not None:
            mskFile = os.path.join(root, rerun, mskFile)
        else:
            mskFile = os.path.join(rerun, mskFile)

    if root is not None:
        imgFile = os.path.join(root, rerun, imgFile)
    else:
        imgFile = os.path.join(rerun, imgFile)

    if (not os.path.isfile(imgFile)) and (not os.path.islink(imgFile)):
        raise Exception("### Can not find the input cutout image : %s !" %
                        imgFile)
    if (not os.path.isfile(mskFile)) and (not os.path.islink(mskFile)):
        raise Exception("### Can not find the input mask image : %s !" %
                        mskFile)

    if os.path.islink(imgFile):
        imgFile = os.readlink(imgFile)
    if os.path.islink(mskFile):
        imgFile = os.readlink(mskFile)

    # Image
    imgHdu = fits.open(imgFile)
    imgArr = imgHdu[0].data
    imgHead = imgHdu[0].header
    # Mask
    mskHdu = fits.open(mskFile)
    mskArr = mskHdu[0].data
    mskHead = mskHdu[0].header

    return imgFile, imgArr, imgHead, mskFile, mskArr, mskHead


def readInputSky(prefix, root=None, rerun='default', rebin='rebin6'):
    """Read sky estimation."""
    skyFile = prefix + '_' + rebin + '_sky.dat'
    if rerun is not None:
        skyFile = os.path.join(root, rerun, skyFile)
    else:
        skyFile = os.path.join(rerun, skyFile)

    if not os.path.isfile(skyFile):
        raise Exception("### Can not find the input sky summary : %s !" %
                        skyFile)

    skySum = open(skyFile, 'r').readlines()
    skyMed = float(skySum[3].split(':')[1].strip())
    skyAvg = float(skySum[4].split(':')[1].strip())
    skyStd = float(skySum[5].split(':')[1].strip())

    return skyMed, skyAvg, skyStd


def getInput1Sersic(config, readinFile='cutout_1ser.in', skyGrad=True,
                    useF1=False, useF4=False):
    """
    Generate the readin file for 1 Sersic GALFIT fitting.

    Parameters:
    """
    f = open(readinFile, 'w')

    f.write('\n')
    f.write('===================================================' +
            '============================\n')
    f.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')
    f.write('A) %s  # Input data image (FITS file)\n' % config['image'][0])
    f.write('B) %s  # Output data image block\n' % config['output'][0])
    f.write('C) %s  # Sigma image name\n' % config['sig'][0])
    f.write('D) %s  # Input PSF image \n' % config['psf'][0])
    f.write('E) 1                   # PSF fine sampling factor' +
            ' relative to data \n')
    f.write('F) %s  # Bad pixel mask\n' % config['mask'][0])
    f.write('G) %s  # File with parameter constraints \n' %
            config['constr'][0])
    f.write('H)  1  %5d  1  %5d  # Image region to fit\n' %
            (config['dimx'], config['dimy']))
    f.write('I) %5d %5d  # Size of the convolution box\n' % (config['convbox'],
            config['convbox']))
    f.write('J) %6.2f  # Magnitude photometric zeropoint \n' % config['zp'])
    f.write('K) %7.3f  %7.3f # Plate scale (dx dy)\n' %
            (config['pix'], config['pix']))
    f.write('O) regular             # Display type (regular, curses, both)\n')
    f.write('P) 0                   # Choose: 0=optimize, 1=model, ' +
            '2=imgblock, 3=subcomps\n')
    f.write('\n')

    f.write('# INITIAL FITTING PARAMETERS\n')
    f.write('#\n')
    f.write('#   For object type, the allowed functions are: \n')
    f.write('#       nuker, sersic, expdisk, devauc, king, psf, ' +
            'gaussian, moffat, \n')
    f.write('#       ferrer, powsersic, sky, and isophote. \n')
    f.write('#  \n')
    f.write('#   Hidden parameters will only appear when they ' +
            'are specified:\n')
    f.write('#       C0 (diskyness/boxyness), \n')
    f.write('#       Fn (n=integer, Azimuthal Fourier Modes),\n')
    f.write('#       R0-R10 (PA rotation, for creating spiral structures).\n')
    f.write('# \n')
    f.write('# -----------------------------------------------------' +
            '------------------------\n')
    f.write('#   par)    par value(s)    fit toggle(s)    # parameter ' +
            'description \n')
    f.write('# -----------------------------------------------------' +
            '------------------------\n')
    f.write('\n')
    f.write('# Object number: 1\n')
    f.write(' 0) sersic    \n')
    f.write(' 1) %7.1f %7.1f  1 1 \n' % (config['x'], config['y']))
    f.write(' 3) %7.3f     1 \n' % config['mag'])
    f.write(' 4) %7.3f     1 \n' % config['re'])
    f.write(' 5) %7.3f     1 \n' % config['nser'])
    f.write(' 6) 0.0000      0          #     ----- \n')
    f.write(' 7) 0.0000      0          #     ----- \n')
    f.write(' 8) 0.0000      0          #     ----- \n')
    f.write(' 9) %7.3f     1 \n' % config['ba'])
    f.write('10) %7.3f     1 \n' % config['pa'])
    if useF1:
        f.write('F1) 0.01 10.00 1 1 ')
    if useF4:
        f.write('F4) 0.01 10.00 1 1 ')
    f.write(' Z) 0                      #  output option ' +
            '(0 = resid., 1 = Dont subtract) \n')
    f.write('\n')
    if config['usesky'] == 1:
        f.write('# Object number: 2\n')
        f.write(' 0) sky                    #  object type\n')
        f.write(' 1) %8.3f  1  #  sky background \n' % config['bkg'])
        if skyGrad:
            f.write(' 2) 0.0000      1          #  dsky/dx \n')
            f.write(' 3) 0.0000      1          #  dsky/dy \n')
        else:
            f.write(' 2) 0.0000      0          #  dsky/dx \n')
            f.write(' 3) 0.0000      0          #  dsky/dy \n')
        f.write(' Z) 1                      #  output option ' +
                '(0 = resid., 1 = Dont subtract) \n')
        f.write('\n')
    f.write('========================================================' +
            '========================\n')
    f.close()


def getInput2Sersic(config, readinFile='cutout_2ser.in', constr=False,
                    skyGrad=True, useF1=False, useF4=False, constrCen=True):
    """Generate the readin file for 2 Sersic GALFIT fitting."""
    f = open(readinFile, 'w')

    loc = os.path.dirname(readinFile)

    f.write('\n')
    f.write('===========================================================' +
            '====================\n')
    f.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')
    f.write('A) %s  # Input data image (FITS file)\n' % config['image'][0])
    f.write('B) %s  # Output data image block\n' % config['output'][0])
    f.write('C) %s  # Sigma image name\n' % config['sig'][0])
    f.write('D) %s  # Input PSF image \n' % config['psf'][0])
    f.write('E) 1                   # PSF fine sampling factor ' +
            'relative to data \n')
    f.write('F) %s  # Bad pixel mask\n' % config['mask'][0])
    if config['constr'][0] == 'None' and constrCen:
        f.write('G) 2comp.cons  # File with parameter constraints \n')
        constrFile = os.path.join(loc, '2comp.cons')
        if not os.path.isfile(constrFile):
            print SEP
            print "### Generate constraint file"
            constrStr = getCenConstrFile(['1', '2'], location=loc)
            print constrStr
            print SEP
    else:
        f.write('G) %s  # File with parameter constraints \n' %
                config['constr'][0])
    f.write('H)  1  %5d  1  %5d  # Image region to fit\n' %
            (config['dimx'], config['dimy']))
    f.write('I) %5d %5d  # Size of the convolution box\n' %
            (config['convbox'], config['convbox']))
    f.write('J) %6.2f  # Magnitude photometric zeropoint \n' % config['zp'])
    f.write('K) %7.3f  %7.3f # Plate scale (dx dy)\n' %
            (config['pix'], config['pix']))
    f.write('O) regular             # Display type (regular, curses, both)\n')
    f.write('P) 0                   # Choose: ' +
            '0=optimize, 1=model, 2=imgblock, 3=subcomps\n')
    f.write('\n')
    f.write('# INITIAL FITTING PARAMETERS\n')
    f.write('#\n')
    f.write('#   For object type, the allowed functions are: \n')
    f.write('#       nuker, sersic, expdisk, devauc, king, ' +
            'psf, gaussian, moffat, \n')
    f.write('#       ferrer, powsersic, sky, and isophote. \n')
    f.write('#  \n')
    f.write('#   Hidden parameters will only appear when ' +
            'they are specified:\n')
    f.write('#       C0 (diskyness/boxyness), \n')
    f.write('#       Fn (n=integer, Azimuthal Fourier Modes),\n')
    f.write('#       R0-R10 (PA rotation, for creating spiral structures).\n')
    f.write('# \n')
    f.write('# -------------------------------------------------------' +
            '----------------------\n')
    f.write('#   par)    par value(s)    fit toggle(s)    ' +
            '# parameter description \n')
    f.write('# -------------------------------------------------------' +
            '----------------------\n')
    f.write('\n')

    f.write('# Object number: 1\n')
    f.write(' 0) sersic    \n')
    f.write(' 1) %7.1f %7.1f  1 1 \n' % (config['x'], config['y']))
    f.write(' 3) %7.3f     1 \n' % (config['mag']+0.6))
    f.write(' 4) %7.3f     1 \n' % (config['re']*0.25))
    f.write(' 5) %7.3f     1 \n' % config['nser'])
    f.write(' 6) 0.0000      0          #     ----- \n')
    f.write(' 7) 0.0000      0          #     ----- \n')
    f.write(' 8) 0.0000      0          #     ----- \n')
    f.write(' 9) %7.3f     1 \n' % config['ba'])
    f.write('10) %7.3f     1 \n' % config['pa'])
    if useF1:
        f.write('F1) 0.01 10.00 1 1 ')
    if useF4:
        f.write('F4) 0.01 10.00 1 1 ')
    f.write(' Z) 0                      #  output option ' +
            '(0 = resid., 1 = Dont subtract) \n')

    f.write('# Object number: 2\n')
    f.write(' 0) sersic    \n')
    f.write(' 1) %7.1f %7.1f  1 1 \n' % (config['x'], config['y']))
    f.write(' 3) %7.3f     1 \n' % (config['mag']+0.8))
    f.write(' 4) %7.3f     1 \n' % (config['re']*1.2))
    f.write(' 5) 0.9    1 \n')
    f.write(' 6) 0.0000      0          #     ----- \n')
    f.write(' 7) 0.0000      0          #     ----- \n')
    f.write(' 8) 0.0000      0          #     ----- \n')
    f.write(' 9) %7.3f     1 \n' % config['ba'])
    f.write('10) %7.3f     1 \n' % config['pa'])
    if useF1:
        f.write('F1) 0.01 10.00 1 1 ')
    if useF4:
        f.write('F4) 0.01 10.00 1 1 ')
    f.write(' Z) 0                      #  output option ' +
            '(0 = resid., 1 = Dont subtract) \n')
    f.write('\n')

    if config['usesky'] == 1:
        f.write('# Object number: 3\n')
        f.write(' 0) sky                    #  object type\n')
        f.write(' 1) %8.3f  1  #  sky background \n' % config['bkg'])
        if skyGrad:
            f.write(' 2) 0.0000      1          #  dsky/dx\n')
            f.write(' 3) 0.0000      1          #  dsky/dy\n')
        else:
            f.write(' 2) 0.0000      0          #  dsky/dx\n')
            f.write(' 3) 0.0000      0          #  dsky/dy\n')
        f.write(' Z) 1                      #  output option ' +
                '(0 = resid., 1 = Dont subtract) \n')
        f.write('\n')
    f.write('=====================================================' +
            '===========================\n')

    f.close()


def getInput3Sersic(config, readinFile='cutout_3ser.in', constr=False,
                    skyGrad=True, useF1=False, useF4=False,
                    constrCen=True):
    """Generate the readin file for 3 Sersic GALFIT fitting."""
    f = open(readinFile, 'w')

    loc = os.path.dirname(readinFile)

    f.write('\n')
    f.write('======================================================' +
            '=========================\n')
    f.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')
    f.write('A) %s  # Input data image (FITS file)\n' % config['image'][0])
    f.write('B) %s  # Output data image block\n' % config['output'][0])
    f.write('C) %s  # Sigma image name\n' % config['sig'][0])
    f.write('D) %s  # Input PSF image \n' % config['psf'][0])
    f.write('E) 1                   # PSF fine sampling factor ' +
            'relative to data \n')
    f.write('F) %s  # Bad pixel mask\n' % config['mask'][0])
    if config['constr'][0] == 'None' and constrCen:
        f.write('G) 3comp.cons  # File with parameter constraints \n')
        constrFile = os.path.join(loc, '3comp.cons')
        if not os.path.isfile(constrFile):
            print SEP
            print " ## Generate comstraint file"
            constrStr = getCenConstrFile(['1', '2', '3'], location=loc)
            print constrStr
            print SEP
    else:
        f.write('G) %s  # File with parameter constraints \n' %
                config['constr'][0])
    f.write('H)  1  %5d  1  %5d  # Image region to fit\n' %
            (config['dimx'], config['dimy']))
    f.write('I) %5d %5d  # Size of the convolution box\n' %
            (config['convbox'], config['convbox']))
    f.write('J) %6.2f  # Magnitude photometric zeropoint \n' % config['zp'])
    f.write('K) %7.3f  %7.3f # Plate scale (dx dy)\n' %
            (config['pix'], config['pix']))
    f.write('O) regular             # Display type (regular, curses, both)\n')
    f.write('P) 0                   # Choose: 0=optimize, ' +
            '1=model, 2=imgblock, 3=subcomps\n')
    f.write('\n')
    f.write('# INITIAL FITTING PARAMETERS\n')
    f.write('#\n')
    f.write('#   For object type, the allowed functions are: \n')
    f.write('#       nuker, sersic, expdisk, devauc, king, psf, ' +
            'gaussian, moffat, \n')
    f.write('#       ferrer, powsersic, sky, and isophote. \n')
    f.write('#  \n')
    f.write('#   Hidden parameters will only appear when ' +
            'they are specified:\n')
    f.write('#       C0 (diskyness/boxyness), \n')
    f.write('#       Fn (n=integer, Azimuthal Fourier Modes),\n')
    f.write('#       R0-R10 (PA rotation, for creating spiral structures).\n')
    f.write('# \n')
    f.write('# -------------------------------------------------------' +
            '----------------------\n')
    f.write('#   par)    par value(s)    fit toggle(s)    ' +
            '# parameter description \n')
    f.write('# -------------------------------------------------------' +
            '----------------------\n')

    f.write('\n')
    f.write('# Object number: 1\n')
    f.write(' 0) sersic    \n')
    f.write(' 1) %7.1f %7.1f  1 1 \n' % (config['x'], config['y']))
    f.write(' 3) %7.3f     1 \n' % (config['mag']+1.2))
    f.write(' 4) %7.3f     1 \n' % (config['re']*0.25))
    f.write(' 5) %7.3f     1 \n' % config['nser'])
    f.write(' 6) 0.0000      0          #     ----- \n')
    f.write(' 7) 0.0000      0          #     ----- \n')
    f.write(' 8) 0.0000      0          #     ----- \n')
    f.write(' 9) %7.3f     1 \n' % config['ba'])
    f.write('10) %7.3f     1 \n' % config['pa'])
    if useF1:
        f.write('F1) 0.01 10.00 1 1 ')
    if useF4:
        f.write('F4) 0.01 10.00 1 1 ')
    f.write(' Z) 0                      #  output option ' +
            '(0 = resid., 1 = Dont subtract) \n')

    f.write('# Object number: 2\n')
    f.write(' 0) sersic    \n')
    f.write(' 1) %7.1f %7.1f  1 1 \n' % (config['x'], config['y']))
    f.write(' 3) %7.3f     1 \n' % (config['mag']+0.9))
    f.write(' 4) %7.3f     1 \n' % (config['re']*0.9))
    f.write(' 5) 0.9    1 \n')
    f.write(' 6) 0.0000      0          #     ----- \n')
    f.write(' 7) 0.0000      0          #     ----- \n')
    f.write(' 8) 0.0000      0          #     ----- \n')
    f.write(' 9) %7.3f     1 \n' % config['ba'])
    f.write('10) %7.3f     1 \n' % config['pa'])
    if useF1:
        f.write('F1) 0.01 10.00 1 1 ')
    if useF4:
        f.write('F4) 0.01 10.00 1 1 ')
    f.write(' Z) 0                      #  output option ' +
            '(0 = resid., 1 = Dont subtract) \n')
    f.write('\n')

    f.write('# Object number: 3\n')
    f.write(' 0) sersic    \n')
    f.write(' 1) %7.1f %7.1f  1 1 \n' % (config['x'], config['y']))
    f.write(' 3) %7.3f     1 \n' % (config['mag']+0.7))
    f.write(' 4) %7.3f     1 \n' % (config['re']*1.3))
    f.write(' 5) 0.5    1 \n')
    f.write(' 6) 0.0000      0          #     ----- \n')
    f.write(' 7) 0.0000      0          #     ----- \n')
    f.write(' 8) 0.0000      0          #     ----- \n')
    f.write(' 9) %7.3f     1 \n' % config['ba'])
    f.write('10) %7.3f     1 \n' % config['pa'])
    if useF1:
        f.write('F1) 0.01 10.00 1 1 ')
    if useF4:
        f.write('F4) 0.01 10.00 1 1 ')
    f.write(' Z) 0                      #  output option ' +
            '(0 = resid., 1 = Dont subtract) \n')
    f.write('\n')

    if config['usesky'] == 1:
        f.write('# Object number: 3\n')
        f.write(' 0) sky                    #  object type\n')
        f.write(' 1) %8.3f  1  #  sky background \n' % config['bkg'])
        if skyGrad:
            f.write(' 2) 0.0000      1          #  dsky/dx\n')
            f.write(' 3) 0.0000      1          #  dsky/dy\n')
        else:
            f.write(' 2) 0.0000      0          #  dsky/dx\n')
            f.write(' 3) 0.0000      0          #  dsky/dy\n')
        f.write(' Z) 1                      #  output option ' +
                '(0 = resid., 1 = Dont subtract) \n')
        f.write('\n')
    f.write('===================================================' +
            '=============================\n')
    f.close()


def coaddCutoutGalfitSimple(prefix, root=None, rerun='default',
                            pix=0.168, useBkg=True,
                            zp=27.0, usePsf=True, galX0=None, galY0=None,
                            galQ0=None, galPA0=None, galRe=None,
                            galSer=2.0, model=None, inFile=None,
                            outFile=None, useSig=True, mag=18.0,
                            constrFile=None, verbose=True,
                            run1=False, run2=False, run3=False,
                            skyGrad=True, ser2Comp=True, ser3Comp=True,
                            useF4=False, useF1=False, imax=150,
                            checkCenter=False, constrCen=True,
                            deleteAfter=False, maskType='mskfin',
                            externalMask=None, abspath=False,
                            show=True, imgSub=False,
                            scale1=0.03, scale2=0.40):
    """
    Run 1-Sersic fitting on HSC cutout image.

    Parameters:
    """
    if verbose:
        print SEP
        print "### Input Image: ", prefix
        print SEP
    """ 0. Organize Input Data """
    # Read in the input image, mask, psf, and their headers
    """ Allow using external mask """
    if externalMask is not None:
        galInput = readSbpInput(prefix, root=root, rerun=rerun,
                                extMsk=externalMask, imgSub=imgSub)
    else:
        galInput = readSbpInput(prefix, root=root, rerun=rerun,
                                maskType=maskType, imgSub=imgSub)
    imgFile, imgArr, imgHead, mskFile, mskArr, mskHead = galInput
    """ Absolute path of the image and mask """
    if abspath:
        imgFile = os.path.abspath(imgFile)
        mskFile = os.path.abspath(mskFile)
        filePath = os.path.dirname(os.path.abspath(mskFile))

    if not imgSameSize(imgArr, mskArr):
        print WAR
        raise Exception("### The Image and Mask need to have " +
                        "EXACTLY same dimensions!")
    dimX, dimY = imgArr.shape

    """ Clean the large image file """
    del imgArr
    del mskArr

    if checkCenter:
        if mskHead['MSK_R20'] == 1:
            print WAR
            raise Exception("### The central region is masked out")

    """ 0a. PSF """
    if usePsf:
        psfFile = prefix + '_psf.fits'
        if root is not None:
            psfFile = os.path.join(root, psfFile)
        if not os.path.isfile(psfFile):
            print WAR
            raise Exception(" XXX Can not find the PSF image : %s", psfFile)
        """ Absolute path of the PSF file"""
        if abspath:
            psfFile = os.path.abspath(psfFile)
    else:
        psfFile = ''

    """ 0b. Sigma Image """
    if useSig:
        sigFile = prefix + '_sig.fits'
        if root is not None:
            sigFile = os.path.join(root, sigFile)
        if not os.path.isfile(sigFile):
            print WAR
            raise Exception(" XXX Can not find the Sigma image : %s", sigFile)
        if abspath:
            sigFile = os.path.abspath(sigFile)
    else:
        sigFile = ''

    """ 0c. Background """
    if useBkg:
        try:
            skyMed, skyAvg, skyStd = readInputSky(prefix, root=root,
                                                  rerun=rerun)
            bkg = skyAvg
            if verbose:
                print "  # Average Background : ", bkg
        except Exception:
            print WAR
            print " XXX CAN NOT FIND THE BACKGROUND DATA !"
            bkg = 0.00
    else:
        bkg = 0.00

    """ 0d. Read-in File """
    if model is None:
        suffix = ''
    else:
        suffix = '_' + model
    if inFile is None:
        inFile = prefix + suffix + '_1ser.in'
        if root is not None:
            inFile = os.path.join(root, rerun, inFile)
        else:
            inFile = os.path.join(rerun, inFile)

    """ 0e. Output File """
    if outFile is None:
        outModel = prefix + suffix + '_1ser.fits'
        if root is not None:
            outFile = os.path.join(root, rerun, outModel)
        else:
            outFile = os.path.join(rerun, outModel)
        if abspath:
            outFile = os.path.join(filePath, outModel)

    """ 0f. Prepare the Input for SBP """
    if (galX0 is None) or (galY0 is None):
        galX, galY = mskHead['GAL_CENX'], mskHead['GAL_CENY']
    else:
        galX, galY = galX0, galY0
    if (galQ0 is None) or (galPA0 is None):
        galQ, galPA = mskHead['GAL_Q'], mskHead['GAL_PA']
    else:
        galQ, galPA = galQ0, galPA0
    galQ = galQ if galQ <= 0.95 else 0.95
    galPA = hUtil.normAngle(galPA, lower=0.0, upper=180.0)

    if galRe is None:
        galR50 = mskHead['GAL_R50']
    else:
        galR50 = galRe

    """ 0g. Convolution Box Size """
    convbox = int(galR50 * 32.0)
    convbox = convbox if convbox >= 600 else 600
    convbox = convbox if convbox <= int(dimX*0.9) else int(dimX*0.9)

    if verbose:
        print SEP
        print " ## Image : ", imgFile
        print " ## Mask  : ", mskFile
        print " ## Sigma : ", sigFile
        print " ## PSF   : ", psfFile
        print " ## galX, galY : ", galX, galY
        print " ## galQ, galPA : ", galQ, galPA
        print " ## galR50 : ", galR50
        print " ## galSer : ", galSer
        print " ## convbox : ", convbox
        print SEP

    """ 0h. Generate the configuration file """
    galfitConfig = np.recarray((1,), dtype=[('x', float), ('y', float),
                               ('ba', float), ('pa', float), ('mag', float),
                               ('re', float), ('nser', float), ('bkg', float),
                               ('image', 'a120'), ('psf', 'a120'),
                               ('mask', 'a120'), ('constr', 'a50'),
                               ('sig', 'a120'), ('pix', float),
                               ('zp', float), ('convbox', int),
                               ('usesky', int), ('dimx', int),
                               ('dimy', int), ('output', 'a120')])
    if useBkg:
        galfitConfig['usesky'] = 1
    else:
        galfitConfig['usesky'] = 0

    galfitConfig['x'] = galX
    galfitConfig['y'] = galX
    galfitConfig['ba'] = galQ
    galfitConfig['pa'] = galPA
    galfitConfig['mag'] = mag
    galfitConfig['re'] = galR50
    galfitConfig['nser'] = galSer
    galfitConfig['bkg'] = bkg
    galfitConfig['image'] = imgFile
    galfitConfig['psf'] = psfFile
    galfitConfig['sig'] = sigFile
    galfitConfig['mask'] = mskFile
    galfitConfig['pix'] = pix
    galfitConfig['zp'] = zp
    galfitConfig['convbox'] = convbox
    galfitConfig['dimx'] = dimX
    galfitConfig['dimy'] = dimY
    galfitConfig['output'] = outFile
    if constrFile is None:
        galfitConfig['constr'] = 'None'
    else:
        galfitConfig['constr'] = constrFile

    """ Directory for the output model """
    if root is not None:
        modRoot = os.path.join(root, rerun)
    else:
        modRoot = rerun

    """ 1a. Generate the Read-in File for 1Ser model"""
    getInput1Sersic(galfitConfig, readinFile=inFile, skyGrad=skyGrad,
                    useF1=useF1, useF4=useF4)
    """ 1b. Execute the GALFIT run """
    if run1:
        done1, plot1 = coaddRunGalfit(inFile, root=modRoot, imax=imax,
                                      zoomSize=int(dimX/2.5),
                                      deleteAfter=deleteAfter,
                                      show=show,
                                      scale1=scale1, scale2=scale2)
        if done1:
            ser1Done = 'DONE'
        else:
            ser1Done = 'FAIL'
        if plot1:
            ser1Plot = 'DONE'
        else:
            ser1Plot = 'FAIL'
    else:
        ser1Done, ser1Plot = 'NRUN', 'NRUN'

    """ Optional: 2-Sersic Model """
    if ser2Comp:
        """ 2a. Generate the Read-in File for 2Ser model"""
        inFile2 = inFile.replace('1ser', '2ser')
        """ 2b. Output File """
        outFile2 = outFile.replace('1ser', '2ser')
        """ 2c. Config """
        config2Ser = copy.deepcopy(galfitConfig)
        config2Ser['output'] = outFile2
        getInput2Sersic(config2Ser, readinFile=inFile2, skyGrad=skyGrad,
                        useF1=useF1, useF4=useF4, constrCen=constrCen)
        """ 2d. Execute the GALFIT run """
        if run2:
            done2, plot2 = coaddRunGalfit(inFile2, root=modRoot, imax=imax,
                                          zoomSize=int(dimX/2.5),
                                          deleteAfter=deleteAfter, show=show,
                                          scale1=scale1, scale2=scale2)
            if done2:
                ser2Done = 'DONE'
            else:
                ser2Done = 'FAIL'
            if plot2:
                ser2Plot = 'DONE'
            else:
                ser2Plot = 'FAIL'
        else:
            ser2Done, ser2Plot = 'NUSE', 'NUSE'
    else:
        ser2Done, ser2Plot = 'NUSE', 'NUSE'

    """ Optional: 3-Sersic Model """
    if ser3Comp:
        """ 3a. Generate the Read-in File for 3Ser model"""
        inFile3 = inFile.replace('1ser', '3ser')
        """ 3b. Output File """
        outFile3 = outFile.replace('1ser', '3ser')
        """ 3c. Config """
        config3Ser = copy.deepcopy(galfitConfig)
        config3Ser['output'] = outFile3
        getInput3Sersic(config3Ser, readinFile=inFile3, skyGrad=skyGrad,
                        useF1=useF1, useF4=useF4, constrCen=constrCen)
        """ 3d. Execute the GALFIT run """
        if run3:
            done3, plot3 = coaddRunGalfit(inFile3, root=modRoot, imax=imax,
                                          zoomSize=int(dimX/2.5),
                                          deleteAfter=deleteAfter, show=show,
                                          scale1=scale1, scale2=scale2)
            if done3:
                ser3Done = 'DONE'
            else:
                ser3Done = 'FAIL'
            if plot3:
                ser3Plot = 'DONE'
            else:
                ser3Plot = 'FAIL'
        else:
            ser3Done, ser3Plot = 'NUSE', 'NUSE'
    else:
        ser3Done, ser3Plot = 'NUSE', 'NUSE'

    return ser1Done, ser1Plot, ser2Done, ser2Plot, ser3Done, ser3Plot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Prefix of the cutout image files")
    parser.add_argument('-r', '--root', dest='root',
                        help='Path to the image files',
                        default=None)
    parser.add_argument('--rerun', dest='rerun',
                        help='Name of the rerun',
                        default='default')

    parser.add_argument('--model', dest='model',
                        help='Suffix of the model',
                        default=None)
    parser.add_argument('--inFile', dest='inFile',
                        help='Name of the read-in file',
                        default=None)
    parser.add_argument('--outFile', dest='outFile',
                        help='Name of the output model',
                        default=None)
    parser.add_argument('--constrFile', dest='constrFile',
                        help='Name of the constraint',
                        default=None)
    parser.add_argument('--externalMask', dest='externalMask',
                        help='External mask file',
                        default=None)
    parser.add_argument('--maskType', dest='maskType',
                        help='Type of the mask use',
                        default=None)
    parser.add_argument('--pix', dest='pix', help='Pixel Scale',
                        type=float, default=0.168)
    parser.add_argument('--zp', dest='zp', help='Photometric zeropoint',
                        type=float, default=27.0)
    parser.add_argument('--imax', dest='imax',
                        help='Maximum number of iterations',
                        type=int, default=150)
    parser.add_argument('--mag', dest='mag', help='Total magnitude',
                        type=float, default=18.00)
    parser.add_argument('--galX0', dest='galX0', help='Galaxy Center: X',
                        type=float, default=None)
    parser.add_argument('--galY0', dest='galY0', help='Galaxy Center: Y',
                        type=float, default=None)
    parser.add_argument('--galQ0', dest='galQ0', help='Axis ratio',
                        type=float, default=None)
    parser.add_argument('--galPA0', dest='galPA0', help='Position angle',
                        type=float, default=None)
    parser.add_argument('--galRe', dest='galRe', help='Effective Radius',
                        type=float, default=None)
    parser.add_argument('--galSer', dest='galSer', help='Sersic Index',
                        type=float, default=2.0)
    parser.add_argument('--noBkg', dest='useBkg', action="store_false",
                        default=True)
    parser.add_argument('--noPsf', dest='usePsf', action="store_false",
                        default=True)
    parser.add_argument('--noSig', dest='useSig', action="store_false",
                        default=True)
    parser.add_argument('--verbose', dest='verbose', action="store_true",
                        default=False)
    parser.add_argument('--run1', dest='run1', action="store_true",
                        default=False)
    parser.add_argument('--run2', dest='run2', action="store_true",
                        default=False)
    parser.add_argument('--run3', dest='run3', action="store_true",
                        default=False)
    parser.add_argument('--ser2Comp', dest='ser2Comp', action="store_true",
                        default=False)
    parser.add_argument('--ser3Comp', dest='ser3Comp', action="store_true",
                        default=False)
    parser.add_argument('--skyGrad', dest='skyGrad', action="store_true",
                        default=False)
    parser.add_argument('--useF1', dest='useF1', action="store_true",
                        default=False)
    parser.add_argument('--useF4', dest='useF4', action="store_true",
                        default=False)
    parser.add_argument('--show', dest='show', action="store_true",
                        default=False)
    parser.add_argument('--noConstrCen', dest='constrCen',
                        action="store_false", default=True)
    parser.add_argument('--noCheckCenter', dest='checkCenter',
                        action="store_false", default=True)
    parser.add_argument('--deleteAfter', dest='deleteAfter',
                        action="store_true", default=False)
    parser.add_argument('--abspath', dest='abspath',
                        action="store_true", default=False)
    parser.add_argument('--imgSub', dest='imgSub',
                        action="store_true", default=False)
    parser.add_argument('--scale1', dest='scale1', help='Scale for image',
                        type=float, default=0.03)
    parser.add_argument('--scale2', dest='scale2', help='Scale for residual',
                        type=float, default=0.40)

    args = parser.parse_args()

    coaddCutoutGalfitSimple(args.prefix, root=args.root, rerun=args.rerun,
                            pix=args.pix,
                            useBkg=args.useBkg, zp=args.zp,
                            usePsf=args.usePsf, galX0=args.galX0,
                            galY0=args.galY0, galQ0=args.galQ0,
                            galPA0=args.galPA0, galRe=args.galRe,
                            galSer=args.galSer, model=args.model,
                            inFile=args.inFile, outFile=args.outFile,
                            useSig=args.useSig, mag=args.mag,
                            constrFile=args.constrFile, verbose=args.verbose,
                            run1=args.run1, run2=args.run2, run3=args.run3,
                            ser2Comp=args.ser2Comp, ser3Comp=args.ser3Comp,
                            skyGrad=args.skyGrad, useF1=args.useF1,
                            useF4=args.useF4, constrCen=args.constrCen,
                            checkCenter=args.checkCenter,
                            deleteAfter=args.deleteAfter,
                            externalMask=args.externalMask,
                            abspath=args.abspath, show=args.show,
                            imgSub=args.imgSub,
                            scale1=args.scale1, scale2=args.scale2)
