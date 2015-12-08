#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import os
import gc
import copy
import argparse
import subprocess
import numpy as np

from scipy.stats import sigmaclip

# Matplotlib default settings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
mpl.rcParams['figure.figsize'] = 12, 10
mpl.rcParams['xtick.major.size'] = 10.0
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.minor.size'] = 5.0
mpl.rcParams['xtick.minor.width'] = 2.5
mpl.rcParams['ytick.major.size'] = 10.0
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['ytick.minor.size'] = 5.0
mpl.rcParams['ytick.minor.width'] = 2.5
mpl.rc('axes', linewidth=3.5)

# Astropy related
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, Column
from pyraf import iraf

# Color table
try:
    import cubehelix  # Cubehelix color scheme
    cmap = cubehelix.cmap(start=0.5, rot=-1.0,
                          minSat=1.2, maxSat=1.2,
                          minLight=0., maxLight=1., gamma=0.5)
except ImportError:
    cmap = 'spectral'
cmap.set_bad('k', 1.)

# Personal
import hscUtils as hUtil

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def correctPositionAngle(ellipOut, paNorm=False):
    """
    Correct the position angle for large jump.

    Parameters:
    """
    if paNorm:
        posAng = ellipOut['pa_norm']
    else:
        posAng = ellipOut['pa']

    for i in range(1, len(posAng)):
        if (posAng[i] - posAng[i - 1]) >= 85.0:
            posAng[i] -= 180.0
        elif (posAng[i] - posAng[i - 1] <= -85.0):
            posAng[i] += 180.0

    if paNorm:
        ellipOut['pa_norm'] = posAng
    else:
        ellipOut['pa'] = posAng

    return ellipOut


def convIso2Ell(ellTab, xpad=0.0, ypad=0.0):
    """
    Convert ellipse results into ellipses for visualization.

    Parameters:
    """
    x = ellTab['x0'] - xpad
    y = ellTab['y0'] - ypad
    pa = ellTab['pa']
    a = ellTab['sma'] * 2.0
    b = ellTab['sma'] * 2.0 * (1.0 - ellTab['ell'])

    ells = [Ellipse(xy=np.array([x[i], y[i]]),
                    width=np.array(b[i]),
                    height=np.array(a[i]),
                    angle=np.array(pa[i]))
            for i in range(x.shape[0])]

    return ells


def maskFits2Pl(inputImage, inputMask):
    """
    Convert the FITS mask into the IRAF .pl file.

    This is stupid....

    Parameters:
    """
    if not os.path.isfile(inputMask):
        raise Exception("Can not find the FITS mask: %s" % inputMask)
    # Name of the .pl mskOri for IRAF
    """
    Why the hell the .pl mask is not working under
    """
    outputMask = inputImage.replace('.fits', '.fits.pl')
    if os.path.isfile(outputMask):
        os.remove(outputMask)
    # Convert the fits format mask into pl format.
    print SEP
    iraf.unlearn('imcopy')
    iraf.imcopy(input=inputMask, output=outputMask, verbose=True)
    print SEP

    return outputMask


def imageMaskNaN(inputImage, inputMask):
    """
    Assigning NaN to mask region.

    Under Pyraf, it seems that .pl file is not working.  This is a work around.
    """
    newImage = inputImage.replace('.fits', '_nan.fits')
    print SEP
    print " ## %s ---> %s " % (inputImage, newImage)
    print SEP

    if os.path.islink(inputImage):
        imgOri = os.readlink(inputImage)
    else:
        imgOri = inputImage
    if not os.path.isfile(imgOri):
        raise Exception("Can not find the FITS image: %s" % imgOri)
    else:
        imgArr = fits.open(imgOri)[0].data
        imgHead = fits.open(imgOri)[0].header

    if os.path.islink(inputMask):
        mskOri = os.readlink(inputMask)
    else:
        mskOri = inputMask
    if not os.path.isfile(mskOri):
        raise Exception("Can not find the FITS mask: %s" % mskOri)
    else:
        mskArr = fits.open(mskOri)[0].data

    imgArr[mskArr > 0] = np.nan

    newHdu = fits.PrimaryHDU(imgArr, header=imgHead)
    hduList = fits.HDUList([newHdu])
    hduList.writeto(newImage, clobber=True)

    del imgArr
    del mskArr

    return newImage


def defaultEllipse(x0, y0, maxsma, ellip0=0.05, pa0=0.0, sma0=6.0, minsma=0.0,
                   linear=False, step=0.08, recenter=True, conver=0.05,
                   hcenter=True, hellip=True, hpa=True, minit=10, maxit=250,
                   olthresh=1.00, mag0=27.0, integrmode='median', usclip=3.0,
                   lsclip=3.0, nclip=2, fflag=0.5, harmonics='none'):
    """
    The default settings for Ellipse.

    Parameters:
    """
    ellipConfig = np.recarray((1,), dtype=[('x0', float), ('y0', float),
                                           ('ellip0', float), ('pa0', float),
                                           ('sma0', float), ('minsma', float),
                                           ('maxsma', float), ('linear', bool),
                                           ('step', float), ('recenter', bool),
                                           ('conver', int), ('hcenter', bool),
                                           ('hellip', bool), ('hpa', bool),
                                           ('minit', int), ('maxit', int),
                                           ('olthresh', float),
                                           ('mag0', float),
                                           ('integrmode', 'a10'),
                                           ('usclip', float),
                                           ('lsclip', float), ('nclip', int),
                                           ('fflag', float),
                                           ('harmonics', 'a10')])
    # Default setting for Ellipse Run
    ellipConfig['x0'] = x0
    ellipConfig['y0'] = y0
    ellipConfig['ellip0'] = ellip0
    ellipConfig['pa0'] = pa0
    ellipConfig['sma0'] = sma0
    ellipConfig['minsma'] = minsma
    ellipConfig['maxsma'] = maxsma
    ellipConfig['linear'] = linear
    ellipConfig['step'] = step
    ellipConfig['recenter'] = recenter
    ellipConfig['conver'] = conver
    ellipConfig['hcenter'] = hcenter
    ellipConfig['hellip'] = hellip
    ellipConfig['hpa'] = hpa
    ellipConfig['minit'] = minit
    ellipConfig['maxit'] = maxit
    ellipConfig['olthresh'] = olthresh
    ellipConfig['mag0'] = mag0
    ellipConfig['integrmode'] = integrmode
    ellipConfig['usclip'] = usclip
    ellipConfig['lsclip'] = lsclip
    ellipConfig['nclip'] = nclip
    ellipConfig['fflag'] = fflag
    ellipConfig['harmonics'] = harmonics

    return ellipConfig


def unlearnEllipse():
    """Unlearn the settings for Ellipse."""
    iraf.unlearn('geompar')
    iraf.unlearn('controlpar')
    iraf.unlearn('samplepar')
    iraf.unlearn('magpar')
    iraf.unlearn('ellipse')


def easierEllipse(ellipConfig, verbose=True,
                  dRad=0.9, dStep=0.03, dFlag=0.05):
    """Make the Ellipse run easier."""
    if verbose:
        print SEP
        print "###  Maxsma %6.1f --> %6.1f" % (ellipConfig['maxsma'],
                                               ellipConfig['maxsma'] *
                                               dRad)
        print "###  Step   %6.2f --> %6.2f" % (ellipConfig['step'],
                                               ellipConfig['step'] +
                                               dStep)
        print "###  Flag   %6.2f --> %6.2f" % (ellipConfig['fflag'],
                                               ellipConfig['fflag'] +
                                               dFlag)
        print SEP
    ellipConfig['maxsma'] *= dRad
    ellipConfig['step'] += dStep
    ellipConfig['fflag'] += dFlag

    return ellipConfig


def setupEllipse(ellipConfig):
    """
    Setup the configuration for Ellipse.

    Parameters:
    """
    cfg = ellipConfig[0]

    # Define parameters for the ellipse run
    # 1. Initial guess of the central X, Y
    if (cfg['x0'] > 0) and (cfg['y0'] > 0):
        iraf.ellipse.x0 = cfg['x0']
        iraf.ellipse.y0 = cfg['y0']
    else:
        raise "Make sure that the input X0 and Y0 are meaningful !", cfg[
            'x0'], cfg['y0']

    # 2. Initial guess of the ellipticity and PA of the first ISOPHOTE
    if (cfg['ellip0'] >= 0.0) and (cfg['ellip0'] < 1.0):
        iraf.ellipse.ellip0 = cfg['ellip0'] if cfg['ellip0'] >= 0.05 else 0.05
    else:
        raise "Make sure that the input Ellipticity is meaningful !", cfg[
            'ellip0']
    if (cfg['pa0'] >= -90.0) and (cfg['pa0'] <= 90.0):
        iraf.ellipse.pa0 = cfg['pa0']
    else:
        raise "Make sure that the input Position Angle is meaningful !", cfg[
            'pa0']

    # 3. Initial radius for ellipse fitting
    iraf.ellipse.sma0 = cfg['sma0']
    # 4. The minimum and maximum radius for the ellipse fitting
    iraf.ellipse.minsma = cfg['minsma']
    iraf.ellipse.maxsma = cfg['maxsma']
    # 5. Parameters about the stepsize during the fitting.
    if cfg['linear']:
        iraf.ellipse.linear = 'yes'
    else:
        iraf.ellipse.linear = 'no'
    iraf.ellipse.geompar.step = cfg['step']
    # 6. Do you want to allow the ellipse to decide the galaxy center during
    # the
    if cfg['recenter']:
        iraf.ellipse.recenter = 'yes'
    else:
        iraf.ellipse.recenter = 'no'
    # 7. The next three parameters control the behavior of the fit
    iraf.ellipse.conver = cfg['conver']
    if cfg['hcenter']:
        iraf.ellipse.hcenter = 'yes'
    else:
        iraf.ellipse.hcenter = 'no'
    if cfg['hellip']:
        iraf.ellipse.hellip = 'yes'
    else:
        iraf.ellipse.hellip = "no"
    if cfg['hpa']:
        iraf.ellipse.hpa = 'yes'
    else:
        iraf.ellipse.hpa = 'no'
    # 8. Parameters about the iterations
    # minit/maxit: minimun and maximum number of the iterations
    iraf.ellipse.minit = cfg['minit']
    iraf.ellipse.maxit = cfg['maxit']
    # 9. Threshold for the object locator algorithm
    iraf.ellipse.olthresh = cfg['olthresh']
    # 10. Make sure the Interactive Mode is turned off
    iraf.ellipse.interactive = 'no'
    # 11. Magnitude Zeropoint
    iraf.ellipse.mag0 = cfg['mag0']
    # 12. Sampler
    intMode = cfg['integrmode']
    intMode = intMode.lower().strip()
    if intMode == 'median':
        iraf.ellipse.integrmode = 'median'
    elif intMode == 'mean':
        iraf.ellipse.integrmode = 'mean'
    elif intMode == 'bi-linear':
        iraf.ellipse.integrmode = 'bi-linear'
    else:
        raise Exception(
            "### Only 'mean', 'median', and 'bi-linear' are available !")
    iraf.ellipse.usclip = cfg['usclip']
    iraf.ellipse.lsclip = cfg['lsclip']
    iraf.ellipse.nclip = cfg['nclip']
    iraf.ellipse.fflag = cfg['fflag']
    # 13. Optional Harmonics
    iraf.ellipse.harmonics = cfg['harmonics']


def ellipRemoveIndef(outTabName, replace='NaN'):
    """
    Remove the Indef values from the Ellipse output.

    Parameters:
    """
    if os.path.exists(outTabName):
        subprocess.call(['sed', '-i_back', 's/INDEF/' +
                        replace + '/g', outTabName])
    else:
        raise Exception('Can not find the input catalog!')

    return outTabName


def readEllipseOut(outTabName, pix=1.0, zp=27.0, exptime=1.0, bkg=0.0,
                   harmonics='none', galR=None, minSma=2.0):
    """
    Read the Ellipse output into a structure.

    Parameters:
    """
    ellipRemoveIndef(outTabName)
    ellipseOut = Table.read(outTabName, format='ascii.no_header')

    # Rename all the columns
    ellipseOut.rename_column('col1',  'sma')
    ellipseOut.rename_column('col2',  'intens')
    ellipseOut.rename_column('col3',  'int_err')
    ellipseOut.rename_column('col4',  'pix_var')
    ellipseOut.rename_column('col5',  'rms')
    ellipseOut.rename_column('col6',  'ell')
    ellipseOut.rename_column('col7',  'ell_err')
    ellipseOut.rename_column('col8',  'pa')
    ellipseOut.rename_column('col9',  'pa_err')
    ellipseOut.rename_column('col10', 'x0')
    ellipseOut.rename_column('col11', 'x0_err')
    ellipseOut.rename_column('col12', 'y0')
    ellipseOut.rename_column('col13', 'y0_err')
    ellipseOut.rename_column('col14', 'grad')
    ellipseOut.rename_column('col15', 'grad_err')
    ellipseOut.rename_column('col16', 'grad_r_err')
    ellipseOut.rename_column('col17', 'rsma')
    ellipseOut.rename_column('col18', 'mag')
    ellipseOut.rename_column('col19', 'mag_lerr')
    ellipseOut.rename_column('col20', 'mag_uerr')
    ellipseOut.rename_column('col21', 'tflux_e')
    ellipseOut.rename_column('col22', 'tflux_c')
    ellipseOut.rename_column('col23', 'tmag_e')
    ellipseOut.rename_column('col24', 'tmag_c')
    ellipseOut.rename_column('col25', 'npix_e')
    ellipseOut.rename_column('col26', 'npix_c')
    ellipseOut.rename_column('col27', 'a3')
    ellipseOut.rename_column('col28', 'a3_err')
    ellipseOut.rename_column('col29', 'b3')
    ellipseOut.rename_column('col30', 'b3_err')
    ellipseOut.rename_column('col31', 'a4')
    ellipseOut.rename_column('col32', 'a4_err')
    ellipseOut.rename_column('col33', 'b4')
    ellipseOut.rename_column('col34', 'b4_err')
    ellipseOut.rename_column('col35', 'ndata')
    ellipseOut.rename_column('col36', 'nflag')
    ellipseOut.rename_column('col37', 'niter')
    ellipseOut.rename_column('col38', 'stop')
    ellipseOut.rename_column('col39', 'a_big')
    ellipseOut.rename_column('col40', 'sarea')
    if harmonics != "none":
        # TODO: Read as many harmonics as necessary
        ellipseOut.rename_column('col41', 'a1')
        ellipseOut.rename_column('col42', 'a1_err')
        ellipseOut.rename_column('col43', 'b1')
        ellipseOut.rename_column('col44', 'b1_err')
        ellipseOut.rename_column('col45', 'a2')
        ellipseOut.rename_column('col46', 'a2_err')
        ellipseOut.rename_column('col47', 'b2')
        ellipseOut.rename_column('col48', 'b2_err')

    # Normalize the PA
    ellipseOut = correctPositionAngle(ellipseOut, paNorm=False)
    ellipseOut.add_column(Column(name='pa_norm',
                          data=np.array([hUtil.normAngle(pa,
                                        lower=-90, upper=90.0, b=True)
                                        for pa in ellipseOut['pa']])))

    # Apply a photometric zeropoint to the magnitude
    ellipseOut['mag'] += zp
    ellipseOut['tmag_e'] += zp
    ellipseOut['tmag_c'] += zp

    # Convert the intensity into surface brightness
    parea = (pix ** 2.0)
    # Surface brightness
    # Fixed the negative intensity
    intensSub = (ellipseOut['intens'] - bkg)
    intensSub[intensSub < 0] = np.nan
    sbp = zp - 2.5 * np.log10(intensSub / (parea * exptime))
    ellipseOut.add_column(Column(name='sbp', data=sbp))
    # Not so accurate estimates of surface brightness error
    sbp_low = zp - 2.5 * np.log10((intensSub + ellipseOut['int_err']) /
                                  (parea * exptime))
    sbp_err = (sbp - sbp_low)
    sbp_upp = (sbp + sbp_err)
    ellipseOut.add_column(Column(name='sbp_low', data=sbp_low))
    ellipseOut.add_column(Column(name='sbp_upp', data=sbp_upp))

    # Convert the unit of radius into arcsecs
    ellipseOut.add_column(Column(name='sma_asec',
                                 data=(ellipseOut['sma'] * pix)))
    ellipseOut.add_column(Column(name='rsma_asec',
                                 data=(ellipseOut['sma'] * pix) ** 0.25))
    # Get the average X0, Y0, Q, and PA
    """
    The choice is pretty random
    """
    if galR is None:
        galR = np.max(ellipseOut['sma']) * 0.18
    avgX, avgY = ellipseGetAvgCen(ellipseOut, galR, minSma=minSma)
    avgQ, avgPA = ellipseGetAvgGeometry(ellipseOut, galR, minSma=minSma)

    ellipseOut.add_column(Column(name='avg_x0',
                                 data=(ellipseOut['sma'] * 0.0 + avgX)))
    ellipseOut.add_column(Column(name='avg_y0',
                                 data=(ellipseOut['sma'] * 0.0 + avgY)))
    ellipseOut.add_column(Column(name='avg_q',
                                 data=(ellipseOut['sma'] * 0.0 + avgQ)))
    ellipseOut.add_column(Column(name='avg_pa',
                                 data=(ellipseOut['sma'] * 0.0 + avgPA)))

    cogOri, maxSma, maxFlux = ellipseGetGrowthCurve(ellipseOut)
    ellipseOut.add_column(Column(name='growth_ori', data=(cogOri)))

    return ellipseOut


def zscale(img, contrast=0.25, samples=500):
    """zScale the image."""
    # Image scaling function form
    # http://hsca.ipmu.jp/hscsphinx/scripts/psfMosaic.html
    ravel = img.ravel()
    if len(ravel) > samples:
        imsort = np.sort(np.random.choice(ravel, size=samples))
    else:
        imsort = np.sort(ravel)

    n = len(imsort)
    idx = np.arange(n)

    med = imsort[n / 2]
    w = 0.25
    i_lo, i_hi = int((0.5 - w) * n), int((0.5 + w) * n)
    p = np.polyfit(idx[i_lo:i_hi], imsort[i_lo:i_hi], 1)
    slope, intercept = p

    z1 = med - (slope / contrast) * (n / 2 - n * w)
    z2 = med + (slope / contrast) * (n / 2 - n * w)

    return z1, z2


def ellipseGetGrowthCurve(ellipOut):
    """
    Extract growth curve from Ellipse output.

    Parameters:
    """
    # The area in unit of pixels covered by an elliptical isophote
    ellArea = np.pi * (ellipOut['sma'] ** 2.0 * (1.0 - ellipOut['ell']))
    # The area in unit covered by the "ring"
    # isoArea = np.append(ellArea[0], [ellArea[1:] - ellArea[:-1]])
    # The total flux inside the "ring"
    isoFlux = np.append(
        ellArea[0], [ellArea[1:] - ellArea[:-1]]) * ellipOut['intens']
    isoTFlux = np.asarray(map(lambda x: np.nansum(isoFlux[0:x + 1]),
                              range(isoFlux.shape[0])))
    # Get the growth curve
    growthCurveOri = np.asarray(isoTFlux)
    indexMax = np.argmax(growthCurveOri)
    maxIsoSma = ellipOut['sma'][indexMax]
    maxIsoFlux = isoTFlux[indexMax]

    return growthCurveOri, maxIsoSma, maxIsoFlux


def ellipseGetR50(ellipseRsma, isoGrowthCurve, simple=True):
    """Estimate R50 from Ellipse output."""
    if len(ellipseRsma) != len(isoGrowthCurve):
        raise "The x and y should have the same size!", (len(ellipseRsma),
                                                         len(isoGrowthCurve))
    else:
        if simple:
            isoRsma50 = ellipseRsma[np.nanargmin(
                np.abs(isoGrowthCurve - 50.0))]
        else:
            isoRsma50 = (np.interp([50.0], isoGrowthCurve, ellipseRsma))[0]

    return isoRsma50


def ellipseGetAvgCen(ellipseOut, outRad, minSma=2.0):
    """Get the Average X0/Y0."""
    try:
        xUse = ellipseOut['x0'][(ellipseOut['sma'] <= outRad) &
                                (np.isfinite(ellipseOut['x0_err'])) &
                                (np.isfinite(ellipseOut['y0_err']))]
        yUse = ellipseOut['y0'][(ellipseOut['sma'] <= outRad) &
                                (np.isfinite(ellipseOut['x0_err'])) &
                                (np.isfinite(ellipseOut['y0_err']))]
        iUse = ellipseOut['intens'][(ellipseOut['sma'] <= outRad) &
                                    (np.isfinite(ellipseOut['x0_err'])) &
                                    (np.isfinite(ellipseOut['y0_err']))]
    except Exception:
        xUse = ellipseOut['x0'][(ellipseOut['sma'] <= outRad)]
        yUse = ellipseOut['y0'][(ellipseOut['sma'] <= outRad)]
        iUse = ellipseOut['intens'][(ellipseOut['sma'] <= outRad)]

    avgCenX = hUtil.numpy_weighted_mean(xUse, weights=iUse)
    avgCenY = hUtil.numpy_weighted_mean(yUse, weights=iUse)

    return avgCenX, avgCenY


def ellipseGetAvgGeometry(ellipseOut, outRad, minSma=2.0):
    """Get the Average Q and PA."""
    tfluxE = ellipseOut['tflux_e']
    ringFlux = np.append(tfluxE[0], [tfluxE[1:] - tfluxE[:-1]])

    try:
        eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad) &
                                 (ellipseOut['sma'] >= minSma) &
                                 (np.isfinite(ellipseOut['ell_err'])) &
                                 (np.isfinite(ellipseOut['pa_err']))]
        pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad) &
                                     (ellipseOut['sma'] >= minSma) &
                                     (np.isfinite(ellipseOut['ell_err'])) &
                                     (np.isfinite(ellipseOut['pa_err']))]
        fUse = ringFlux[(ellipseOut['sma'] <= outRad) &
                        (ellipseOut['sma'] >= minSma) &
                        (np.isfinite(ellipseOut['ell_err'])) &
                        (np.isfinite(ellipseOut['pa_err']))]
    except Exception:
        try:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad) &
                                     (ellipseOut['sma'] >= 0.5) &
                                     (np.isfinite(ellipseOut['ell_err'])) &
                                     (np.isfinite(ellipseOut['pa_err']))]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad) &
                                         (ellipseOut['sma'] >= 0.5) &
                                         (np.isfinite(ellipseOut['ell_err'])) &
                                         (np.isfinite(ellipseOut['pa_err']))]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad) &
                            (ellipseOut['sma'] >= 0.5) &
                            (np.isfinite(ellipseOut['ell_err'])) &
                            (np.isfinite(ellipseOut['pa_err']))]
        except Exception:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad) &
                                     (ellipseOut['sma'] >= 0.5)]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad) &
                                         (ellipseOut['sma'] >= 0.5)]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad) &
                            (ellipseOut['sma'] >= 0.5)]

    avgQ = 1.0 - hUtil.numpy_weighted_mean(eUse, weights=fUse)
    avgPA = hUtil.numpy_weighted_mean(pUse, weights=fUse)

    return avgQ, avgPA


def ellipseFixNegIntens(ellipseOut, value=1e-6):
    """Replace the negative value from the intensity."""
    ellipseNew = copy.deepcopy(ellipseOut)
    ellipseNew['intens'][ellipseNew['intens'] < 0.0] = value

    return ellipseNew


def ellipseGetOuterBoundary(ellipseOut, ratio=1.2, margin=0.2, polyOrder=12,
                            median=False, threshold=None):
    """Get the outer boundary of the output 1-D profile."""
    try:
        meanErr = np.nanmean(ellipseOut['int_err'])
        if threshold is not None:
            thre = threshold
        else:
            thre = meanErr
        negRad = ellipseOut['rsma'][np.where(ellipseOut['intens'] <= thre)]

        if (negRad is np.nan) or (len(negRad) < 3):
            try:
                uppIntens = np.nanmax(ellipseOut['intens']) * 0.01
                indexUse = np.where(ellipseOut['intens'] <= uppIntens)
            except Exception:
                uppIntens = np.nanmax(ellipseOut['intens']) * 0.03
                indexUse = np.where(ellipseOut['intens'] <= uppIntens)
            intensFit = hUtil.polyFit(ellipseOut['rsma'][indexUse],
                                      ellipseOut['intens'][indexUse],
                                      order=polyOrder)
            radUse = ellipseOut['rsma'][indexUse]
            negRad = radUse[np.where(intensFit <= meanErr)]

        if median:
            outRsma = np.nanmedian(negRad)
        else:
            outRsma = np.nanmean(negRad)
        return (outRsma ** 4.0) * ratio
    except Exception, errMsg:
        print str(errMsg)
        return None


def ellipsePlotSummary(ellipOut, image, maxRad=None, mask=None, radMode='rsma',
                       outPng='ellipse_summary.png', zp=27.0, threshold=None,
                       showZoom=False, useZscale=True):
    """
    Make a summary plot of the ellipse run.

    Parameters:
    """
    """ Left side: SBP """
    reg1 = [0.07, 0.05, 0.455, 0.35]
    reg2 = [0.07, 0.40, 0.455, 0.15]
    reg3 = [0.07, 0.55, 0.455, 0.15]
    reg4 = [0.07, 0.70, 0.455, 0.15]
    reg5 = [0.07, 0.85, 0.455, 0.14]

    """ Right side: Curve of growth & IsoMap """
    reg6 = [0.59, 0.05, 0.39, 0.30]
    reg7 = [0.59, 0.35, 0.39, 0.16]
    reg8 = [0.59, 0.55, 0.39, 0.39]

    fig = plt.figure(figsize=(20, 20))
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
    if useZscale:
        imin, imax = zscale(imgMsk, contrast=0.6, samples=500)
    else:
        imin, imax = np.nanmin(imgMsk), np.nanmax(imgMsk)
    if mask is not None:
        msk = fits.open(mask)[0].data
        imgMsk[msk > 0] = np.nan

    """ Find the proper outer boundary """
    sma = ellipOut['sma']
    radOuter = ellipseGetOuterBoundary(
        ellipOut, ratio=1.2, threshold=threshold)
    if not np.isfinite(radOuter):
        print WAR
        print " XX  radOuter is NaN, use 0.80 * max(SMA) instead !"
        print WAR
        radOuter = np.nanmax(sma) * 0.80

    indexUse = np.where(ellipOut['sma'] <= (radOuter * 1.2))
    print SEP
    print "###     OutRadius : ", radOuter

    """ Get growth curve """
    curveOri = ellipOut['growth_ori']
    curveOri[curveOri < 0] = np.nan
    curveCor = ellipOut['growth_cor']
    curveCor[curveCor < 0] = np.nan
    growthCurveOri = -2.5 * np.log10(curveOri) + zp
    growthCurveNew = -2.5 * np.log10(curveCor) + zp

    maxIsoFluxOri = np.nanmax(ellipOut['growth_ori'][indexUse])
    magFluxOri100 = -2.5 * np.log10(maxIsoFluxOri) + zp
    print "###     MagTot OLD : ", magFluxOri100

    maxIsoFlux = np.nanmax(ellipOut['growth_cor'][indexUse])
    magFlux50 = -2.5 * np.log10(maxIsoFlux * 0.50) + zp
    magFlux100 = -2.5 * np.log10(maxIsoFlux) + zp
    print "###     MagTot NEW : ", magFlux100

    indMaxFlux = np.nanargmax(ellipOut['growth_cor'][indexUse])
    maxIsoSbp = ellipOut['sbp_upp'][indMaxFlux]
    print "###     MaxIsoSbp : ", maxIsoSbp
    print SEP

    """ Type of Radius """
    if radMode is 'rsma':
        rad = ellipOut['rsma']
        radStr = 'RSMA (pixel$^{1/4}$)'
        minRad = 0.41 if 0.41 >= np.nanmin(
            ellipOut['rsma']) else np.nanmin(ellipOut['rsma'])
        imgR50 = (imgX / 2.0) ** 0.25
        radOut = (radOuter * 1.2) ** 0.25
        radOut = radOut if radOut <= imgR50 else imgR50
        if maxRad is None:
            maxRad = np.nanmax(rad)
            maxSma = np.nanmax(ellipOut['sma'])
        else:
            maxSma = maxRad
            maxRad = maxRad ** 0.25
    elif radMode is 'sma':
        rad = ellipOut['sma']
        radStr = 'SMA (pixel)'
        minRad = 0.05 if 0.05 >= np.nanmin(
            ellipOut['sma']) else np.nanmin(ellipOut['sma'])
        imgR50 = (imgX / 2.0)
        radOut = (radOuter * 1.2)
        radOut = radOut if radOut <= imgR50 else imgR50
        if maxRad is None:
            maxRad = maxSma = np.nanmax(rad)
        else:
            maxSma = maxRad
    elif radMode is 'log':
        rad = np.log10(ellipOut['sma'])
        radStr = 'log (SMA/pixel)'
        minRad = 0.01 if 0.01 >= np.log10(
            np.nanmin(ellipOut['sma'])) else np.log10(
            np.nanmin(ellipOut['sma']))
        imgR50 = np.log10(imgX / 2.0)
        radOut = np.log10(radOuter * 1.2)
        radOut = radOut if radOut <= imgR50 else imgR50
        if maxRad is None:
            maxRad = np.nanmax(rad)
            maxSma = np.nanmax(ellipOut['sma'])
        else:
            maxSma = maxRad
            maxRad = np.log10(maxRad)
    else:
        raise Exception('### Wrong type of Radius: sma, rsma, log')

    """ ax1 SBP """
    ax1.minorticks_on()
    ax1.invert_yaxis()
    ax1.tick_params(axis='both', which='major', labelsize=22, pad=8)

    ax1.set_xlabel(radStr, fontsize=23)
    ax1.set_ylabel('${\mu}$ (mag/arcsec$^2$)', fontsize=28)

    ax1.fill_between(rad[indexUse], ellipOut['sbp_upp'][indexUse],
                     ellipOut['sbp_low'][indexUse], facecolor='k', alpha=0.3)

    ax1.plot(rad[indexUse], ellipOut['sbp'][
             indexUse], '-', color='r', linewidth=3.0)

    ax1.set_xlim(minRad, radOut)
    sbpBuffer = 0.5
    minSbp = np.nanmin(ellipOut['sbp_low'][indexUse]) - sbpBuffer
    maxSbp = maxIsoSbp + 1.1
    maxSbp = maxSbp if maxSbp >= 29.0 else 28.9
    maxSbp = maxSbp if maxSbp <= 32.0 else 31.9

    ax1.set_ylim(maxSbp, minSbp)

    """ ax2 Ellipticity """
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax2.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax2.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax2.locator_params(axis='y', tight=True, nbins=4)

    ax2.set_ylabel('$e$', fontsize=30)

    print SEP
    print "###     AvgEll", (1.0 - ellipOut['avg_q'][0])
    ax2.axhline((1.0 - ellipOut['avg_q'][0]),
                color='k', linestyle='--', linewidth=2)

    ax2.fill_between(rad[indexUse],
                     ellipOut['ell'][indexUse] + ellipOut['ell_err'][indexUse],
                     ellipOut['ell'][indexUse] - ellipOut['ell_err'][indexUse],
                     facecolor='r', alpha=0.25)
    ax2.plot(rad[indexUse], ellipOut['ell'][
             indexUse], '-', color='r', linewidth=2.0)

    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2.set_xlim(minRad, radOut)
    ellBuffer = 0.02
    minEll = np.nanmin(ellipOut['ell'][indexUse] -
                       ellipOut['ell_err'][indexUse])
    maxEll = np.nanmax(ellipOut['ell'][indexUse] +
                       ellipOut['ell_err'][indexUse])
    ax2.set_ylim(minEll - ellBuffer, maxEll + ellBuffer)

    """ ax3 PA """
    ax3.minorticks_on()
    ax3.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax3.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax3.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax3.locator_params(axis='y', tight=True, nbins=4)

    ax3.set_ylabel('PA (degree)',  fontsize=23)

    medPA = np.nanmedian(ellipOut['pa_norm'][indexUse])
    avgPA = ellipOut['avg_pa'][0]
    if (avgPA - medPA >= 85.0) and (avgPA <= 92.0):
        avgPA -= 180.0
    elif (avgPA - medPA <= -85.0) and (avgPA >= -92.0):
        avgPA += 180.0

    print "###     AvgPA", avgPA
    ax3.axhline(avgPA, color='k', linestyle='--', linewidth=3.0)

    ax3.fill_between(rad[indexUse],
                     ellipOut['pa_norm'][indexUse] +
                     ellipOut['pa_err'][indexUse],
                     ellipOut['pa_norm'][indexUse] -
                     ellipOut['pa_err'][indexUse],
                     facecolor='r', alpha=0.25)
    ax3.plot(rad[indexUse], ellipOut['pa_norm'][
             indexUse], '-', color='r', linewidth=2.0)

    ax3.xaxis.set_major_formatter(NullFormatter())
    ax3.set_xlim(minRad, radOut)
    paBuffer = 4.0
    minPA = np.nanmin(ellipOut['pa_norm'][indexUse] -
                      ellipOut['pa_err'][indexUse])
    maxPA = np.nanmax(ellipOut['pa_norm'][indexUse] +
                      ellipOut['pa_err'][indexUse])
    minPA = minPA if minPA >= -110.0 else -100.0
    maxPA = maxPA if maxPA <= 110.0 else 100.0
    ax3.set_ylim(minPA - paBuffer, maxPA + paBuffer)

    """ ax4 X0/Y0 """
    ax4.minorticks_on()
    ax4.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax4.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax4.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax4.locator_params(axis='y', tight=True, nbins=4)

    ax4.set_ylabel('X0 or Y0 (pixel)', fontsize=23)

    print "###     AvgX0", ellipOut['avg_x0'][0]
    print "###     AvgY0", ellipOut['avg_y0'][0]
    ax4.axhline(ellipOut['avg_x0'][0], linestyle='--',
                color='r', alpha=0.6, linewidth=3.0)
    ax4.fill_between(rad[indexUse],
                     ellipOut['x0'][indexUse] + ellipOut['x0_err'][indexUse],
                     ellipOut['x0'][indexUse] - ellipOut['x0_err'][indexUse],
                     facecolor='r', alpha=0.25)
    ax4.plot(rad[indexUse], ellipOut['x0'][indexUse], '-', color='r',
             linewidth=2.0, label='X0')

    ax4.axhline(ellipOut['avg_y0'][0], linestyle='--',
                color='b', alpha=0.6, linewidth=3.0)
    ax4.fill_between(rad[indexUse],
                     ellipOut['y0'][indexUse] + ellipOut['y0_err'][indexUse],
                     ellipOut['y0'][indexUse] - ellipOut['y0_err'][indexUse],
                     facecolor='b', alpha=0.25)
    ax4.plot(rad[indexUse], ellipOut['y0'][indexUse], '-', color='b',
             linewidth=2.0, label='Y0')

    ax4.xaxis.set_major_formatter(NullFormatter())
    ax4.set_xlim(minRad, radOut)
    xBuffer = 3.0
    minX0 = np.nanmin(ellipOut['x0'][indexUse])
    maxX0 = np.nanmax(ellipOut['x0'][indexUse])
    minY0 = np.nanmin(ellipOut['y0'][indexUse])
    maxY0 = np.nanmax(ellipOut['y0'][indexUse])
    minCen = minX0 if minX0 <= minY0 else minY0
    maxCen = maxX0 if maxX0 >= maxY0 else maxY0
    ax4.set_ylim(minCen - xBuffer, maxCen + xBuffer)

    """ ax5 A4/B4 """
    ax5.minorticks_on()
    ax5.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax5.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax5.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax5.locator_params(axis='y', tight=True, nbins=4)

    ax5.set_ylabel('A4 or B4',  fontsize=23)

    ax5.axhline(0.0, linestyle='-', color='k', alpha=0.3)
    ax5.fill_between(rad[indexUse],
                     ellipOut['a4'][indexUse] + ellipOut['a4_err'][indexUse],
                     ellipOut['a4'][indexUse] - ellipOut['a4_err'][indexUse],
                     facecolor='r', alpha=0.25)
    ax5.plot(rad[indexUse], ellipOut['a4'][indexUse], '-', color='r',
             linewidth=2.0, label='A4')

    ax5.fill_between(rad[indexUse],
                     ellipOut['b4'][indexUse] + ellipOut['b4_err'][indexUse],
                     ellipOut['b4'][indexUse] - ellipOut['b4_err'][indexUse],
                     facecolor='b', alpha=0.25)
    ax5.plot(rad[indexUse], ellipOut['b4'][indexUse], '-', color='b',
             linewidth=2.0, label='B4')

    ax5.xaxis.set_major_formatter(NullFormatter())
    ax5.set_xlim(minRad, radOut)

    abBuffer = 0.02
    minA4 = np.nanmin(ellipOut['a4'][indexUse])
    minB4 = np.nanmin(ellipOut['b4'][indexUse])
    maxA4 = np.nanmax(ellipOut['a4'][indexUse])
    maxB4 = np.nanmax(ellipOut['b4'][indexUse])
    minAB = minA4 if minA4 <= minB4 else minB4
    maxAB = maxA4 if maxA4 >= maxB4 else maxB4
    ax5.set_ylim(minAB - abBuffer, maxAB + abBuffer)

    """ ax6 Growth Curve """
    ax6.minorticks_on()
    ax6.tick_params(axis='both', which='major', labelsize=22, pad=8)
    ax6.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax6.yaxis.set_major_locator(MaxNLocator(prune='upper'))

    ax6.set_xlabel(radStr, fontsize=23)
    ax6.set_ylabel('Curve of Growth (mag)', fontsize=24)

    ax1.text(0.6, 0.85, 'mag$_{tot}=%5.2f$' % magFlux100, fontsize=24,
             transform=ax1.transAxes)

    ax6.axhline(magFlux100, linestyle='-', color='k', alpha=0.5, linewidth=2,
                label='mag$_{100}$')
    ax6.axhline(magFlux50,  linestyle='--', color='k', alpha=0.5, linewidth=2,
                label='mag$_{50}$')
    ax6.axvline(imgR50,  linestyle='-', color='g', alpha=0.4, linewidth=2.5)

    ax6.plot(rad, growthCurveOri, '--', color='g', linewidth=3.5,
             label='curve$_{old}$')
    """ XXXX """
    ax6.plot(rad, growthCurveNew, '-', color='r', linewidth=3.5,
             label='curve$_{new}$')

    ax6.axvline(radOut, linestyle='--', color='b', alpha=0.8, linewidth=3.0)
    ax6.legend(loc=[0.35, 0.50], fontsize=24)

    ax6.set_xlim(minRad, maxRad)

    """ ax7 Intensity Curve """
    ax7.minorticks_on()
    ax7.tick_params(axis='both', which='major', labelsize=22, pad=10)
    ax7.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax7.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax7.locator_params(axis='y', tight=True, nbins=4)

    ax7.axhline(0.0, linestyle='-', color='k', alpha=0.5, linewidth=2.5)
    ax7.fill_between(rad, ellipOut['intens'] + ellipOut['int_err'],
                     ellipOut['intens'] - ellipOut['int_err'],
                     facecolor='b', alpha=0.2)
    ax7.plot(rad, ellipOut['intens'], '-', color='b', linewidth=3.0)
    ax7.axvline(imgR50,  linestyle='-', color='g', alpha=0.4, linewidth=2.5)

    """ TODO: Could be problematic """
    indexOut = np.where(ellipOut['intens'] <= (
        0.002 * np.nanmax(ellipOut['intens'])))

    ax7.xaxis.set_major_formatter(NullFormatter())
    ax7.set_xlim(minRad, maxRad)
    minOut = np.nanmin(ellipOut['intens'][indexOut] -
                       ellipOut['int_err'][indexOut])
    maxOut = np.nanmax(ellipOut['intens'][indexOut] +
                       ellipOut['int_err'][indexOut])
    sepOut = (maxOut - minOut) / 10.0
    minY = (minOut - sepOut) if (minOut - sepOut) >= 0.0 else (-1.0 * sepOut)
    ax7.set_ylim(minY, maxOut)

    ax7.axvline(radOut, linestyle='--', color='b', alpha=0.8, linewidth=3.0)

    """ ax8 IsoPlot """

    imgFile = os.path.basename(image)
    imgTitle = imgFile.replace('.fits', '')
    imgTitle = imgTitle.replace('_img', '')

    ax8.tick_params(axis='both', which='major', labelsize=20)
    ax8.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax8.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax8.set_title(imgTitle, fontsize=25, fontweight='bold')
    ax8.title.set_position((0.5, 1.05))

    galX0 = ellipOut['avg_x0'][0]
    galY0 = ellipOut['avg_y0'][0]
    imgSizeX, imgSizeY = img.shape
    if (galX0 > maxSma) and (galY0 > maxSma) and showZoom:
        zoomReg = imgMsk[np.int(galX0 - maxSma):np.int(galX0 + maxSma),
                         np.int(galY0 - maxSma):np.int(galY0 + maxSma)]
        # Define the new center of the cropped images
        xPad = (imgSizeX / 2.0 - maxSma)
        yPad = (imgSizeY / 2.0 - maxSma)
    else:
        zoomReg = imgMsk
        xPad = 0
        yPad = 0

    # Show the image
    ax8.imshow(np.arcsinh(zoomReg), interpolation="none",
               vmin=imin, vmax=imax, cmap=cmap, origin='lower')
    # Get the Shapes
    ellipIso = convIso2Ell(ellipOut, xpad=xPad, ypad=yPad)

    # Overlay the ellipses on the image
    for e in ellipIso:
        ax8.add_artist(e)
        e.set_clip_box(ax8.bbox)
        e.set_alpha(0.9)
        e.set_edgecolor('r')
        e.set_facecolor('none')
        e.set_linewidth(1.5)
    """ Save Figure """
    fig.savefig(outPng)
    plt.close(fig)

    print SEP

    return


def saveEllipOut(ellipOut, prefix, ellipCfg=None, verbose=True):
    """
    Save the Ellipse output to file.

    Parameters:
    """
    outPkl = prefix + '.pkl'
    outCfg = prefix + '.cfg'
    outCsv = prefix + '.csv'

    """ Save a Pickle file """
    hUtil.saveToPickle(ellipOut, outPkl)
    if os.path.isfile(outPkl):
        if verbose:
            print SEP
            print "###     Save Ellipse output to .pkl file: %s" % outPkl
    else:
        raise Exception("### Something is wrong with the .pkl file")

    """ Save a .CSV file """
    ascii.write(ellipOut, outCsv, format='csv')
    if os.path.isfile(outCsv):
        if verbose:
            print "###     Save Ellipse output to .csv file: %s" % outCsv
    else:
        raise Exception("### Something is wrong with the .csv file")

    """ Save the current configuration to a .pkl file """
    if ellipCfg is not None:
        hUtil.saveToPickle(ellipCfg, outCfg)
        if os.path.isfile(outCfg):
            if verbose:
                print "###     Save configuration to a .cfg file: %s" % outCfg
                print SEP
        else:
            raise Exception("### Something is wrong with the .pkl file")


def galSBP(image, mask=None, galX=None, galY=None, inEllip=None,
           maxSma=None, iniSma=6.0, galR=20.0, galQ=0.9, galPA=0.0,
           pix=0.168, bkg=0.00, stage=3, minSma=0.0,
           gain=3.0, expTime=1.0, zpPhoto=27.0,
           maxTry=4, minIt=20, maxIt=200,
           ellipStep=0.12, uppClip=3.0, lowClip=3.0,
           nClip=2, fracBad=0.5, intMode="mean",
           plMask=True, conver=0.05, recenter=True,
           verbose=True, linearStep=False, saveOut=True, savePng=True,
           olthresh=0.5, harmonics='1 2', outerThreshold=None,
           updateIntens=False, psfSma=6.0, suffix='', useZscale=True):
    """
    Running Ellipse to Extract 1-D profile.

    stage  = 1: All Free
            2: Center Fixed
            3: All geometry fixd
            4: Force Photometry, must have inEllip
    :returns: TODO
    """
    gc.collect()
    verStr = 'yes' if verbose else 'no'
    """ Check input files """
    if os.path.islink(image):
        imgOri = os.readlink(image)
    else:
        imgOri = image

    if not os.path.isfile(imgOri):
        raise Exception("### Can not find the input image: %s !" % imgOri)

    """ Conver the .fits mask to .pl file if necessary """
    if mask is not None:
        if os.path.islink(mask):
            mskOri = os.readlink(mask)
        else:
            mskOri = mask

        if not os.path.isfile(mskOri):
            raise Exception("### Can not find the input mask: %s !" % mskOri)

        if plMask:
            print COM
            print "###  Will use the *.pl Mask"
            print COM
            plFile = maskFits2Pl(imgOri, mask)
            if not os.path.isfile(plFile):
                raise Exception("### Can not find the .pl mask: %s !" % plFile)
            imageUse = imgOri
        else:
            print COM
            print "###  Will use the *nan.fits Mask"
            print COM
            imageNew = imageMaskNaN(imgOri, mask)
            if not os.path.isfile(imageNew):
                raise Exception(
                    "### Can not find the NaN-Masked image: %s" % imageNew)
            imageUse = imageNew
    else:
        imageUse = imgOri
        mskOri = None

    """ Estimate the maxSMA if none is provided """
    if (maxSma is None) or (galX is None) or (galY is None):
        data = (fits.open(imgOri))[0].data
        dimX, dimY = data.shape
        imgSize = dimX if (dimX >= dimY) else dimY
        imgR = (imgSize / 2.0)
        imgX = dimX / 2.0
        imgY = dimY / 2.0
        if maxSma is None:
            maxSma = imgR * 1.6
        if galX is None:
            galX = imgX
        if galY is None:
            galY = imgY

    iniSma = iniSma if iniSma >= 5.0 else 5.0
    if verbose:
        print SEP
        print "###      galX, galY : ", galX, galY
        print "###      galR : ", galR
        print "###      iniSma, maxSma : ", iniSma, maxSma
        print "###      Stage : ", stage
        print "###      Step : ", ellipStep
        print SEP

    """ Check the stage """
    if stage == 1:
        hcenter, hellip, hpa = False, False, False
    elif stage == 2:
        hcenter, hellip, hpa = True, False, False
    elif stage == 3:
        hcenter, hellip, hpa = True, True, True
    elif stage == 4:
        hcenter, hellip, hpa = True, True, True
        if (inEllip is None) or (not os.path.isfile(inEllip)):
            raise Exception(
                "### Can not find the input ellip file: %s !" % inEllip)
    else:
        raise Exception("### Available step: 1 , 2 , 3 , 4")

    """ Get the default Ellipse settings """
    if verbose:
        print SEP
        print "##       Set up the Ellipse configuration"
        print SEP
    galEll = (1.0 - galQ)
    ellipCfg = defaultEllipse(galX, galY, maxSma, ellip0=galEll, pa0=galPA,
                              sma0=iniSma, minsma=minSma, linear=linearStep,
                              step=ellipStep, recenter=recenter,
                              conver=conver, hcenter=hcenter, hellip=hellip,
                              hpa=hpa, minit=minIt, maxit=maxIt,
                              olthresh=olthresh, mag0=zpPhoto,
                              integrmode=intMode, usclip=uppClip,
                              lsclip=lowClip, nclip=nClip, fflag=fracBad,
                              harmonics=harmonics)
    """ Name of the output files """
    if suffix == '':
        suffix = '_ellip_' + suffix + str(stage).strip()
    elif suffix[-1] != '_':
        suffix = '_ellip_' + suffix + '_' + str(stage).strip()
    else:
        suffix = '_ellip_' + suffix + str(stage).strip()
    outBin = image.replace('.fits', suffix + '.bin')
    outTab = image.replace('.fits', suffix + '.tab')
    outCdf = image.replace('.fits', suffix + '.cdf')

    """ Call the STSDAS.ANALYSIS.ISOPHOTE package """
    if verbose:
        print '\n' + SEP
        print "##       Call STSDAS.ANALYSIS.ISOPHOTE() "
        print SEP
    iraf.stsdas()
    iraf.analysis()
    iraf.isophote()

    """ Start the Ellipse Run """
    attempts = 0
    while attempts < maxTry:
        if verbose:
            print '\n' + SEP
            print "##       Start the Ellipse Run: Attempt ", (attempts + 1)
            print SEP
        try:
            """ Config the parameters for ellipse """
            unlearnEllipse()
            setupEllipse(ellipCfg)

            """ Ellipse run """
            # Check and remove outputs from the previous Ellipse run
            if os.path.exists(outBin):
                os.remove(outBin)
            # Start the Ellipse fitting
            print SEP
            print "###      Input Image   : %s" % imageUse
            print "###      Output Binary : %s" % outBin
            if stage != 4:
                iraf.ellipse(input=imageUse, output=outBin, verbose=verStr)
            else:
                print "###      Input Binary  : %s" % inEllip
                iraf.ellipse(input=imageUse, output=outBin, inellip=inEllip,
                             verbose=verStr)
            print SEP
            # Check if the Ellipse run is finished
            if not os.path.isfile(outBin):
                raise Exception("XXX Ellipse not done !")
            else:
                # Remove the existed .tab and .cdf file
                if os.path.isfile(outTab):
                    os.remove(outTab)
                if os.path.isfile(outCdf):
                    os.remove(outCdf)
                # Tdump the .bin table into a .tab file
                iraf.unlearn('tdump')
                iraf.tdump.columns = ''
                iraf.tdump(outBin, datafil=outTab, cdfile=outCdf)

                # Read in the Ellipse output tab
                ellipOut = readEllipseOut(outTab, zp=zpPhoto, pix=pix,
                                          exptime=expTime, bkg=bkg,
                                          harmonics=harmonics, minSma=psfSma)
                radOuter = ellipseGetOuterBoundary(ellipOut, ratio=1.2)
                sma = ellipOut['sma']
                if radOuter is None:
                    # XXX : Don't quit yet
                    # raise Exception("### Error : Can not decide radOuter ")
                    print WAR
                    print "XXX  radOuter is NaN, use 0.75 * max(SMA) instead !"
                    print WAR
                    radOuter = np.nanmax(sma) * 0.75

                """ Update the intensity """
                if updateIntens:
                    indexBkg = np.where(ellipOut['sma'] > radOuter * 1.2)
                    if indexBkg[0].shape[0] > 0:
                        try:
                            intensBkg = ellipOut['intens'][indexBkg]
                            clipArr, clipL, clipU = sigmaclip(intensBkg,
                                                              2.0, 2.0)
                            avgBkg = np.nanmean(clipArr)
                            if not np.isfinite(avgBkg):
                                avgBkg = 0.0
                        except Exception:
                            avgBkg = 0.0
                    else:
                        avgBkg = 0.0
                else:
                    avgBkg = 0.0
                print "###     Average Outer Intensity : ", avgBkg
                print SEP

                ellipOut['intens'] -= avgBkg
                ellipOut.add_column(Column(name='avg_bkg',
                                    data=(sma * 0.0 + avgBkg)))

                """ Update the curve of growth """
                cog1,  mm, ff = ellipseGetGrowthCurve(ellipOut)
                ellipOut.add_column(Column(name='growth_cor', data=(cog1)))

                """ Update the outer radius """
                radOuter = ellipseGetOuterBoundary(ellipOut, ratio=1.2)
                if not np.isfinite(radOuter):
                    # XXX : Don't just quit
                    # raise Exception("### Error : Can not decide radOuter ")
                    print " XXX radOuter is NaN, use 0.80 * max(SMA) instead !"
                    radOuter = np.nanmax(sma) * 0.80
                ellipOut.add_column(
                    Column(name='rad_outer', data=(sma * 0.0 + radOuter)))

                """ Update the total magnitude """
                indexUse = np.where(ellipOut['sma'] <= (radOuter * 1.2))
                maxIsoFlux = np.nanmax(ellipOut['growth_cor'][indexUse])
                magFlux100 = -2.5 * np.log10(maxIsoFlux) + zpPhoto
                ellipOut.add_column(
                    Column(name='mag_tot', data=(sma * 0.0 + magFlux100)))

                if savePng:
                    outPng = image.replace('.fits', suffix + '.png')
                    ellipsePlotSummary(ellipOut, imgOri, maxRad=None,
                                       mask=mskOri, outPng=outPng,
                                       threshold=outerThreshold,
                                       useZscale=useZscale)
                if saveOut:
                    outPre = image.replace('.fits', suffix)
                    saveEllipOut(ellipOut, outPre, ellipCfg=ellipCfg,
                                 verbose=verbose)

                gc.collect()
                break
        except Exception as error:
            attempts += 1
            print WAR
            print "###  Error Information : ", error
            print "###  !!! Make the Ellipse Run A Little Bit Easier !"
            print WAR
            ellipCfg = easierEllipse(ellipCfg)

        if not os.path.isfile(outBin):
            ellipOut = None
            print WAR
            print "###  ELLIPSE RUN FAILED AFTER %3d ATTEMPTS!!!" % maxTry
            print WAR

        gc.collect()

    return ellipOut


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Name of the input image")
    parser.add_argument("--suffix",
                        help="Suffix of the output files",
                        default='')
    parser.add_argument("--mask", dest='mask',
                        help="Name of the input mask",
                        default=None)
    parser.add_argument("--intMode", dest='intMode',
                        help="Method for integration",
                        default='mean')
    parser.add_argument('--x0', dest='galX',
                        help='Galaxy center in X-dimension',
                        type=float, default=None)
    parser.add_argument('--y0', dest='galY',
                        help='Galaxy center in Y-dimension',
                        type=float, default=None)
    parser.add_argument('--inEllip', dest='inEllip',
                        help='Input Ellipse table',
                        default=None)
    parser.add_argument('--minSma', dest='minSma',
                        help='Minimum radius for Ellipse Run',
                        type=float, default=0.0)
    parser.add_argument('--maxSma', dest='maxSma',
                        help='Maximum radius for Ellipse Run',
                        type=float, default=None)
    parser.add_argument('--iniSma', dest='iniSma',
                        help='Initial radius for Ellipse Run',
                        type=float, default=10.0)
    parser.add_argument('--galR', dest='galR',
                        help='Typical size of the galaxy',
                        type=float, default=20.0)
    parser.add_argument('--galQ', dest='galQ',
                        help='Typical axis ratio of the galaxy',
                        type=float, default=0.9)
    parser.add_argument('--galPA', dest='galPA',
                        help='Typical PA of the galaxy',
                        type=float, default=0.0)
    parser.add_argument('--stage', dest='stage',
                        help='Stage of Ellipse Run',
                        type=int, default=3, choices=range(1, 4))
    parser.add_argument('--pix', dest='pix',
                        help='Pixel Scale',
                        type=float, default=0.168)
    parser.add_argument('--bkg', dest='bkg',
                        help='Background level',
                        type=float, default=0.0)
    parser.add_argument('--step', dest='step',
                        help='Step size',
                        type=float, default=0.12)
    parser.add_argument('--uppClip', dest='uppClip',
                        help='Upper limit for clipping',
                        type=float, default=3.0)
    parser.add_argument('--lowClip', dest='lowClip',
                        help='Upper limit for clipping',
                        type=float, default=3.0)
    parser.add_argument('--nClip', dest='nClip',
                        help='Upper limit for clipping',
                        type=int, default=2)
    parser.add_argument('--olthresh', dest='olthresh',
                        help='Central locator threshold',
                        type=float, default=0.30)
    parser.add_argument('--zpPhoto', dest='zpPhoto',
                        help='Photometric zeropoint',
                        type=float, default=27.0)
    parser.add_argument('--outThre', dest='outerThreshold',
                        help='Outer threshold',
                        type=float, default=None)
    parser.add_argument('--fracBad', dest='fracBad',
                        help='Outer threshold',
                        type=float, default=0.5)
    parser.add_argument('--maxTry', dest='maxTry',
                        help='Maximum number of ellipse run',
                        type=int, default=4)
    parser.add_argument('--minIt', dest='minIt',
                        help='Minimum number of iterations',
                        type=int, default=20)
    parser.add_argument('--maxIt', dest='maxIt',
                        help='Maximum number of iterations',
                        type=int, default=150)
    parser.add_argument('--plot', dest='plot', action="store_true",
                        help='Generate summary plot', default=True)
    parser.add_argument('--verbose', dest='verbose', action="store_true",
                        default=True)
    parser.add_argument('--linear', dest='linear', action="store_true",
                        default=False)
    parser.add_argument('--save', dest='save', action="store_true",
                        default=True)
    parser.add_argument('--plmask', dest='plmask', action="store_true",
                        default=True)
    parser.add_argument('--updateIntens', dest='updateIntens',
                        action="store_true", default=True)

    args = parser.parse_args()

    galSBP(args.image, mask=args.mask,
           galX=args.galX, galY=args.galY,
           inEllip=args.inEllip,
           maxSma=args.maxSma,
           iniSma=args.iniSma,
           galR=args.galR,
           galQ=args.galQ,
           galPA=args.galPA,
           pix=args.pix,
           bkg=args.bkg,
           stage=args.stage,
           minSma=args.minSma,
           gain=3.0,
           expTime=1.0,
           zpPhoto=args.zpPhoto,
           maxTry=args.maxTry,
           minIt=args.minIt,
           maxIt=args.maxIt,
           ellipStep=args.step,
           uppClip=args.uppClip,
           lowClip=args.lowClip,
           nClip=args.nClip,
           fracBad=args.fracBad,
           intMode=args.intMode,
           suffix=args.suffix,
           plMask=args.plmask,
           conver=0.05,
           recenter=True,
           verbose=args.verbose,
           linearStep=args.linear,
           saveOut=args.save,
           savePng=args.plot,
           olthresh=args.olthresh,
           harmonics='none',
           outerThreshold=args.outerThreshold,
           updateIntens=args.updateIntens)
