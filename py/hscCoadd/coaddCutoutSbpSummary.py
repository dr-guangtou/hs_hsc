#!/usr/bin/env python
# encoding: utf-8
"""Summarize the SBP profiles of a HSC galaxy."""

from __future__ import (division, print_function)

import os
import copy
import fnmatch
import warnings
import argparse

import numpy as np
import cPickle as pickle
from scipy.interpolate import interp1d

# Astropy
# from astropy.io import fits
# from astropy.stats import sigma_clip
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar

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

# Personal
# import galSBP
import hscUtils as hUtil

"""
Absolute magnitude of the Sun in HSC filters
Right now, just use the DES filters
"""
SUN_G = 5.08
SUN_R = 4.62
SUN_I = 4.52
SUN_Z = 4.52
SUN_Y = 4.51
"""
Extinction correction factor for HSC
A\_lambda = Coeff * E(B-V)
"""
A_G = 3.233
A_R = 2.291
A_I = 1.635
A_Z = 1.261
A_Y = 1.076
"""
List of HSC Filters.
"""
HSC_FILTERS = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
"""
Common radius array
"""
RSMA_COMMON = np.arange(0.4, 4.2, 0.01)
EMPTY = (RSMA_COMMON * np.nan)
"""
For output
"""
COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def sbpCompare(galTab, sumPng, pngSize=10):
    """QA plot of the SBPs."""
    reg1 = [0.1, 0.1, 0.88, 0.88]
    fig = plt.figure(figsize=(pngSize, pngSize))
    ax1 = fig.add_axes(reg1)

    """ ax1 SBP """
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', labelsize=20, pad=8)

    radStr = 'RSMA (kpc$^{1/4}$)'
    ax1.set_xlabel(radStr, fontsize=23)
    ax1.set_ylabel('${\mu}$ ($L_{\odot}$/kpc$^2$)', fontsize=26)

    if galTab.meta['MU_I4']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muI4'],
                 linestyle=':',
                 c='r', linewidth=3.0, alpha=0.7,
                 label='muI4')

    if galTab.meta['MU_I5']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muI5'],
                 linestyle=':',
                 c='r', linewidth=3.0, alpha=0.7,
                 label='muI5')

    if galTab.meta['MU_I6']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muI6'],
                 linestyle=':',
                 c='r', linewidth=3.0, alpha=0.7,
                 label='muI6')

    if galTab.meta['MU_I2']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muI2'],
                 linestyle='--',
                 c='r', linewidth=3.5, alpha=0.8,
                 label='muI2')

    if galTab.meta['MU_I3']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muI3'],
                 linestyle='-.',
                 c='r', linewidth=3.5, alpha=0.8,
                 label='muI3')

    ax1.plot(galTab['rKpc'] ** 0.25, galTab['muI1'],
             linestyle='-',
             c='r', linewidth=3.0, alpha=0.9,
             label='muI1')

    if galTab.meta['MU_G2']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muG2'],
                 linestyle='--',
                 c='b', linewidth=3.5, alpha=0.8,
                 label='muG2')

    if galTab.meta['MU_G3']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muG3'],
                 linestyle='-.',
                 c='b', linewidth=3.5, alpha=0.8,
                 label='muG3')

    if galTab.meta['MU_G1']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muG1'],
                 linestyle='-',
                 c='b', linewidth=3.0, alpha=0.9,
                 label='muG1')

    if galTab.meta['MU_R2']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muR2'],
                 linestyle='--',
                 c='g', linewidth=3.5, alpha=0.8,
                 label='muR2')

    if galTab.meta['MU_R3']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muR3'],
                 linestyle='-.',
                 c='g', linewidth=3.5, alpha=0.8,
                 label='muR3')

    if galTab.meta['MU_R1']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muR1'],
                 linestyle='-',
                 c='g', linewidth=3.0, alpha=0.9,
                 label='muR1')

    if galTab.meta['MU_Z2']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muZ2'],
                 linestyle='--',
                 c='m', linewidth=3.5, alpha=0.8,
                 label='muZ2')

    if galTab.meta['MU_Z3']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muZ3'],
                 linestyle='-.',
                 c='m', linewidth=3.5, alpha=0.8,
                 label='muZ3')

    if galTab.meta['MU_Z1']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muZ1'],
                 linestyle='-',
                 c='m', linewidth=3.0, alpha=0.9,
                 label='muZ1')

    if galTab.meta['MU_Y2']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muY2'],
                 linestyle='--',
                 c='k', linewidth=3.5, alpha=0.8,
                 label='muY2')

    if galTab.meta['MU_Y3']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muY3'],
                 linestyle='-.',
                 c='k', linewidth=3.5, alpha=0.8,
                 label='muY3')

    if galTab.meta['MU_Y1']:
        ax1.plot(galTab['rKpc'] ** 0.25, galTab['muY1'],
                 linestyle='-',
                 c='k', linewidth=3.0, alpha=0.9,
                 label='muY1')

    ax1.set_xlim(0.45, 4.4)
    ax1.set_ylim(3.01, 9.99)

    ax1.legend(loc=[0.83, 0.5], fontsize=11)

    if galTab.meta['CONTAM']:
        ax1.text(0.25, 0.25, 'CONTAMINATED', fontsize=20,
                 transform=ax1.transAxes)

    """ Save Figure """
    fig.savefig(sumPng, dpi=80)
    plt.close(fig)

    return


def interpSbp(rad, data, radCommon=RSMA_COMMON, kind='slinear'):
    """
    Interpolate 1-D SBP data.

    Parameters:
        kind  :
    """
    if len(rad) != len(data):
        raise Exception("# The radius and data should be the same in size!")
    intrpFunc = interp1d(rad, data, kind=kind, bounds_error=False)

    return intrpFunc(radCommon)


def toColorArr(data, bottom=None, top=None):
    """
    Convert a data array to "color array" (between 0 and 1).

    Parameters:
        bottom, top  :
    """
    if top is not None:
        data[data >= top] = top
    if bottom is not None:
        data[data <= bottom] = bottom

    return ((data - np.nanmin(data)) /
            (np.nanmax(data) - np.nanmin(data))) * 255.0


def errAdd(err1, err2):
    """Add error quadral..."""
    return np.sqrt((err1 ** 2.0) +
                   (err2 ** 2.0))


def logAdd(para1, para2):
    """Useful for adding magnitudes."""
    return np.log10((10.0 ** np.asarray(para1)) +
                    (10.0 ** np.asarray(para2)))


def pixKpc(redshift, pix=0.168, show=True, npix=1.0):
    """
    Get the corresponding Kpc size of a pixel.

    Parameters:
    """
    pixKpc = pix * npix * hUtil.cosmoScale(redshift)

    if show:
        print("# %d pixel(s) = %6.3f Kpc" % (npix, pixKpc))

    return pixKpc


def normProf(sma, sbp, minSma, maxSma):
    """
    Naive method to normalize the profile.

    Parameters:
        sbp    : Array for surface brightness profile
        sma    : Radius range
        minSma : Minimum SMA
        maxSma   Maximum SMA
    """
    offset = np.nanmedian(sbp[(sma >= minSma) &
                              (sma <= maxSma)])
    return (sbp-offset)


def getDimming(z1, z0=0.1, factor=3.0):
    """
    Get the surface brightness dimming effect.

    Parameters:
        z1: Observed redshift
        z0: Reference redshift
            Default = 0.1
    """
    return (factor * np.log10((1.0 + z1) / (1.0 + z0)))


def getLuminosity(mag, redshift, extinction=None,
                  amag_sun=None):
    """Get the absolute magnitude or luminosity."""
    distmod = hUtil.cosmoDistMod(redshift)
    absMag = (mag - distmod)
    if extinction is not None:
        absMag -= extinction
    if amag_sun is not None:
        absMag = ((amag_sun - absMag) / 2.5)

    return absMag


def getExtinction(ra, dec, a_lambda=None):
    """
    Estimate the Galactic extinction for HSC filters.

    Parameters:
        ra, dec : The input coordinates can be arrays
    """
    # Check the input, if it's scalar, convert to array
    if not hasattr(ra, "__len__") or not hasattr(dec, "__len__"):
        ra = np.asrray([ra])
        dec = np.asarray([dec])
    if len(ra) != len(dec):
        raise Exception("## The RA and DEC should contain " +
                        "same number of elements!")
    # First try mwdust from Jo Bovy
    try:
        import mwdust
        sfd = mwdust.SFD(sf10=True)

        from astropy.coordinates import SkyCoord
        coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
        galactic = coords.galactic
        l, b = galactic.l, galactic.b
        ebv = sfd(l, b, 0)
    except ImportError:
        try:
            # Then try sncosmo
            from sncosmo import SFD98Map
            dustDir = os.environ.get('DUST_DIR')
            if (not os.path.isfile(os.path.join(dustDir,
                                   'SFD_dust_4096_ngp.fits'))) or (
                not os.path.isfile(os.path.join(dustDir,
                                   'SFD_dust_4096_sgp.fits'))):
                print('# DUST_DIR : %s' % dustDir)
                raise Exception("# Can not find the SFD dust map!")
            else:
                sfd = SFD98Map(dustDir)
                ebv = sfd.get_ebv((ra, dec))
        except ImportError:
            raise Exception("# Both mwdust and sncosmo are not available")
    if a_lambda is not None:
        return (ebv * a_lambda)
    else:
        return ebv


def readProfile(ellFile):
    """Load the pickle format 1-D profile."""
    if os.path.isfile(ellFile):
        return pickle.load(open(ellFile, 'rb'))
    else:
        print(WAR)
        print("!!! Can not find the Ellipse Output at %s" % ellFile)
        return None


def findProfile(pattern, loc, verbose=False):
    """
    Find the prefix of the ellipse profiles.

    Parameters:
    """
    result = []
    for root, dirs, files in os.walk(loc):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    if verbose:
        print("### %d files found !" % len(result))

    return result


def getEllipProfile(galid, base, prefix, model, psf=False,
                    filter='HSC-I', rerun='default',
                    verbose=False, imgSub=True):
    """
    Find and load the Ellipse output.

    Parameters:
    """
    galid = str(galid).strip()
    location = os.path.join(base, galid, filter, rerun)
    if psf:
        ellType = 'psf'
    elif imgSub:
        ellType = 'imgsub'
    else:
        ellType = 'img'
    """Ellipse result file name"""
    ellFile = (prefix + '_' + str(galid) + '_' + filter +
               '_full_' + ellType + '_ellip_' + model + '.pkl')
    ellFile = os.path.join(location, ellFile)

    if not os.path.isfile(ellFile):
        if verbose:
            print(WAR)
            print('!!! Can not find the Ellipse profile ! %s' % ellFile)
        return None
    else:
        ellProf = readProfile(ellFile)
        return ellProf


def geomExtract(loc, galID, redshift, filter,
                prefix, rerun, model, imgSub=True,
                verbose=False, interp=True):
    """
    Return geometric information.

    Parameters:
    """
    prof = getEllipProfile(galID, loc, prefix, model,
                           filter=filter, rerun=rerun,
                           verbose=verbose, imgSub=imgSub)
    if prof is not None:
        """ Get physical pixel scale and distant module """
        scale = hUtil.cosmoScale(redshift)
        """ Convert unit of major axis radius to Kpc """
        sma_kpc = prof['sma_asec'] * scale
        rsma_kpc = (sma_kpc ** 0.25)
        """ Basic geometric information """
        ell = prof['ell']
        ell_err = prof['ell_err']
        pa = prof['pa']
        pa_err = prof['pa_err']
        if not interp:
            return sma_kpc, ell, ell_err, pa, pa_err
        else:
            ell_i = interpSbp(rsma_kpc, ell,
                              radCommon=RSMA_COMMON,
                              kind='slinear')
            ell_err_i = interpSbp(rsma_kpc, ell_err,
                                  radCommon=RSMA_COMMON,
                                  kind='slinear')
            pa_i = interpSbp(rsma_kpc, pa,
                             radCommon=RSMA_COMMON,
                             kind='slinear')
            pa_err_i = interpSbp(rsma_kpc, pa_err,
                                 radCommon=RSMA_COMMON,
                                 kind='slinear')
            sma_common = (RSMA_COMMON ** 4.0)

            return sma_common, ell_i, ell_err_i, pa_i, pa_err_i
    else:
        return None


def sbpExtract(loc, galID, redshift, filter,
               prefix, rerun, model,
               zp=27.0, extinction=0.0, imgSub=True,
               amag_sun=None, m2l=None, psf=False,
               origin=False, verbose=False, interp=True):
    """
    Return important SBP information.

    Parameters:
    """
    prof = getEllipProfile(galID, loc, prefix, model,
                           filter=filter, rerun=rerun,
                           verbose=verbose, psf=psf,
                           imgSub=imgSub)
    if prof is not None:
        ell = correctProf(prof, redshift,
                          extinction=extinction,
                          zp=zp, amag_sun=amag_sun,
                          dimming=True, corCurve=True,
                          verbose=verbose, m2l=m2l,
                          interp=interp)
        if origin:
            return ell, prof
        else:
            return ell
    else:
        return None


def sbpCollect(loc, prefix, galID, redshift, rerun='default',
               a_g=0.0, a_r=0.0, a_i=0.0,
               a_z=0.0, a_y=0.0, suffix=None,
               m2l_g=None, m2l_r=None, m2l_i=None,
               m2l_z=None, m2l_y=None, imgSub=True,
               verbose=False, save=True, interp=True,
               sumFolder='sbp_sum', sample=None):
    """
    Collect profiles from the cutout folder.

    This is a quick work-around, eventually should us a Class for dataset

    This works for the redBCG and nonBCG datasets 2015-12-09
    And, can not use for any profile with suffix
    """
    """Location and Table name"""
    sumDir = os.path.join(loc, sumFolder)
    if not os.path.isdir(sumDir):
        os.mkdir(sumDir)

    """Name of the summary table of each galaxy"""
    if sample is None:
        strTemp = str(galID)
    else:
        strTemp = str(sample).strip() + '_' + str(galID)
    if suffix is None:
        sumTab = strTemp + '_sbp_sum.fits'
    else:
        sumTab = strTemp + '_sbp_sum_' + suffix + '.fits'
    """If the directory does not exist, make it"""
    if not os.path.exists(sumDir):
        os.mkdir(sumDir)
    sumTable = os.path.join(sumDir, sumTab)

    """
    The basic reference model
    """
    refEllI = sbpExtract(loc, galID, redshift, 'HSC-I',
                         prefix, rerun, 'default_3',
                         extinction=a_i, m2l=m2l_i,
                         amag_sun=SUN_I, verbose=verbose,
                         interp=interp, imgSub=imgSub)

    if refEllI is not None:
        """ Reference profile in I-band """
        rad, muI1, lumI1, errI1 = refEllI

        """ Create a NaN array """
        if interp:
            empty = EMPTY
        else:
            empty = copy.deepcopy(rad)
            empty[:] = np.nan

        """
        I-band Geometry
        """
        refGeomI = geomExtract(loc, galID, redshift, 'HSC-I',
                               prefix, rerun, 'default_2',
                               verbose=verbose, interp=interp,
                               imgSub=imgSub)
        if refGeomI is not None:
            r, ell, ellErr, pa, paErr = refGeomI
            isGeom = True
        else:
            ell, ellErr, pa, paErr = empty, empty, empty, empty
            isGeom = False
            if verbose:
                print(WAR)
                print('### Can not find the geometry profile ' +
                      'for I-band : %s!' % str(galID))

        """
        I PSF
        """
        psfI = sbpExtract(loc, galID, redshift,
                          'HSC-I', prefix, rerun, '3',
                          m2l=m2l_i, extinction=0.0,
                          amag_sun=SUN_I, verbose=verbose,
                          interp=interp, psf=True)
        if psfI is not None:
            r, psfMuI, temp1, temp2 = psfI
            isPsfI = True
        else:
            psfMuI = empty
            isPsfI = False
            if verbose:
                print(WAR)
                print('### Can not find the PSF SBP ' +
                      'for I-band : %s!' % str(galID))

        """
        I largeR1
        """
        ellI2 = sbpExtract(loc, galID, redshift,
                           'HSC-I', prefix, rerun, 'multi1_4',
                           m2l=m2l_i, extinction=a_i,
                           amag_sun=SUN_I, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellI2 is not None:
            r, muI2, lumI2, errI2 = ellI2
            isMuI2 = True
        else:
            muI2, lumI2, errI2 = empty, empty, empty
            isMuI2 = False
            if verbose:
                print(WAR)
                print('### Can not find the small mask SBP ' +
                      'for I-band : %s!' % str(galID))

        """
        I smallR1
        """
        ellI3 = sbpExtract(loc, galID, redshift,
                           'HSC-I', prefix, rerun, 'multi2_4',
                           m2l=m2l_i, extinction=a_i,
                           amag_sun=SUN_I, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellI3 is not None:
            r, muI3, lumI3, errI3 = ellI3
            isMuI3 = True
        else:
            muI3, lumI3, errI3 = empty, empty, empty
            isMuI3 = False
            if verbose:
                print(WAR)
                print('### Can not find the large mask SBP ' +
                      'for I-band : %s!' % str(galID))

        """
        Compare the default SBP with the largeMask one, if large difference
        is detected within the inner ~50 Kpc, set a Flag to indicate it.
        """
        radThreshold = 50.0   # Kpc
        diffThreshold = 0.4   # mag / arcsec^2
        if isMuI3:
            diffMuI = np.abs(muI3 - muI1)
            if np.nanmax(diffMuI[rad <= radThreshold]) > diffThreshold:
                contaminated = True
                if verbose:
                    print("## %s is contaminated by nearby object!" % galID)
            else:
                contaminated = False
        else:
            print("## WARNING: MU_I3 is not available; Set CONTAMINATED=False")
            contaminated = False

        """
        I multi3
        """
        ellI4 = sbpExtract(loc, galID, redshift,
                           'HSC-I', prefix, rerun, 'multi3_3',
                           m2l=m2l_i, extinction=a_i,
                           amag_sun=SUN_I, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellI4 is not None:
            r, muI4, lumI4, errI4 = ellI4
            isMuI4 = True
        else:
            muI4, lumI4, errI4 = empty, empty, empty
            isMuI4 = False
            if verbose:
                print(WAR)
                print('### Can not find the multi3 SBP ' +
                      'for I-band : %s!' % str(galID))

        """
        I multi4
        """
        ellI5 = sbpExtract(loc, galID, redshift,
                           'HSC-I', prefix, rerun, 'multi4_3',
                           m2l=m2l_i, extinction=a_i,
                           amag_sun=SUN_I, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellI5 is not None:
            r, muI5, lumI5, errI5 = ellI5
            isMuI5 = True
        else:
            muI5, lumI5, errI5 = empty, empty, empty
            isMuI5 = False
            if verbose:
                print(WAR)
                print('### Can not find the multi4 SBP ' +
                      'for I-band : %s!' % str(galID))

        """
        I multi5
        """
        ellI6 = sbpExtract(loc, galID, redshift,
                           'HSC-I', prefix, rerun, 'multi5_3',
                           m2l=m2l_i, extinction=a_i,
                           amag_sun=SUN_I, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellI6 is not None:
            r, muI6, lumI6, errI6 = ellI6
            isMuI6 = True
        else:
            muI6, lumI6, errI6 = empty, empty, empty
            isMuI6 = False
            if verbose:
                print(WAR)
                print('### Can not find the multi5 SBP ' +
                      'for I-band : %s!' % str(galID))

        """
        G default
        """
        ellG1 = sbpExtract(loc, galID, redshift,
                           'HSC-G', prefix, rerun, 'default_4',
                           m2l=m2l_g, extinction=a_g,
                           amag_sun=SUN_G, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellG1 is not None:
            r, muG1, lumG1, errG1 = ellG1
            isMuG1 = True
        else:
            muG1, lumG1, errG1 = empty, empty, empty
            isMuG1 = False
            if verbose:
                print(WAR)
                print('### Can not find the default SBP ' +
                      'for G-band : %s!' % str(galID))

        """
        G PSF
        """
        psfG = sbpExtract(loc, galID, redshift,
                          'HSC-G', prefix, rerun, '3',
                          m2l=m2l_g, extinction=0.0,
                          amag_sun=SUN_G, verbose=verbose,
                          interp=interp, psf=True)
        if psfG is not None:
            r, psfMuG, temp1, temp2 = psfG
            isPsfG = True
        else:
            psfMuG = empty
            isPsfG = False
            if verbose:
                print(WAR)
                print('### Can not find the PSF SBP ' +
                      'for G-band : %s!' % str(galID))

        """
        G small mask
        """
        ellG2 = sbpExtract(loc, galID, redshift,
                           'HSC-G', prefix, rerun, 'default_msksmall_4',
                           m2l=m2l_g, extinction=a_g,
                           amag_sun=SUN_G, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellG2 is not None:
            r, muG2, lumG2, errG2 = ellG2
            isMuG2 = True
        else:
            muG2, lumG2, errG2 = empty, empty, empty
            isMuG2 = False
            if verbose:
                print(WAR)
                print('### Can not find the small mask SBP ' +
                      'for G-band : %s!' % str(galID))

        """
        G large mask
        """
        ellG3 = sbpExtract(loc, galID, redshift,
                           'HSC-G', prefix, rerun, 'default_msklarge_4',
                           m2l=m2l_g, extinction=a_g,
                           amag_sun=SUN_G, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellG3 is not None:
            r, muG3, lumG3, errG3 = ellG3
            isMuG3 = True
        else:
            muG3, lumG3, errG3 = empty, empty, empty
            isMuG3 = False
            if verbose:
                print(WAR)
                print('### Can not find the large mask SBP ' +
                      'for G-band : %s!' % str(galID))

        """
        R default
        """
        ellR1 = sbpExtract(loc, galID, redshift,
                           'HSC-R', prefix, rerun, 'default_4',
                           m2l=m2l_r, extinction=a_r,
                           amag_sun=SUN_R, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellR1 is not None:
            r, muR1, lumR1, errR1 = ellR1
            isMuR1 = True
        else:
            muR1, lumR1, errR1 = empty, empty, empty
            isMuR1 = False
            if verbose:
                print(WAR)
                print('### Can not find the default SBP ' +
                      'for R-band : %s!' % str(galID))

        """
        R PSF
        """
        psfR = sbpExtract(loc, galID, redshift,
                          'HSC-R', prefix, rerun, '3',
                          m2l=m2l_r, extinction=0.0,
                          amag_sun=SUN_R, verbose=verbose,
                          interp=interp, psf=True)
        if psfR is not None:
            r, psfMuR, temp1, temp2 = psfR
            isPsfR = True
        else:
            psfMuR = empty
            isPsfR = False
            if verbose:
                print(WAR)
                print('### Can not find the PSF SBP ' +
                      'for R-band : %s!' % str(galID))

        """
        R small mask
        """
        ellR2 = sbpExtract(loc, galID, redshift,
                           'HSC-R', prefix, rerun, 'default_msksmall_4',
                           m2l=m2l_r, extinction=a_r,
                           amag_sun=SUN_R, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellR2 is not None:
            r, muR2, lumR2, errR2 = ellR2
            isMuR2 = True
        else:
            muR2, lumR2, errR2 = empty, empty, empty
            isMuR2 = False
            if verbose:
                print(WAR)
                print('### Can not find the small mask SBP ' +
                      'for R-band : %s!' % str(galID))

        """
        R large mask
        """
        ellR3 = sbpExtract(loc, galID, redshift,
                           'HSC-R', prefix, rerun, 'default_msklarge_4',
                           m2l=m2l_r, extinction=a_r,
                           amag_sun=SUN_R, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellR3 is not None:
            r, muR3, lumR3, errR3 = ellR3
            isMuR3 = True
        else:
            muR3, lumR3, errR3 = empty, empty, empty
            isMuR3 = False
            if verbose:
                print(WAR)
                print('### Can not find the large mask SBP ' +
                      'for R-band : %s!' % str(galID))

        """
        Z default
        """
        ellZ1 = sbpExtract(loc, galID, redshift,
                           'HSC-Z', prefix, rerun, 'default_4',
                           m2l=m2l_z, extinction=a_z,
                           amag_sun=SUN_Z, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellZ1 is not None:
            r, muZ1, lumZ1, errZ1 = ellZ1
            isMuZ1 = True
        else:
            muZ1, lumZ1, errZ1 = empty, empty, empty
            isMuZ1 = False
            if verbose:
                print(WAR)
                print('### Can not find the default SBP ' +
                      'for Z-band : %s!' % str(galID))

        """
        Z PSF
        """
        psfZ = sbpExtract(loc, galID, redshift,
                          'HSC-Z', prefix, rerun, '3',
                          m2l=m2l_z, extinction=0.0,
                          amag_sun=SUN_Z, verbose=verbose,
                          interp=interp, psf=True)
        if psfZ is not None:
            r, psfMuZ, temp1, temp2 = psfZ
            isPsfZ = True
        else:
            psfMuZ = empty
            isPsfZ = False
            if verbose:
                print(WAR)
                print('### Can not find the PSF SBP ' +
                      'for Z-band : %s!' % str(galID))

        """
        Z small mask
        """
        ellZ2 = sbpExtract(loc, galID, redshift,
                           'HSC-Z', prefix, rerun, 'default_msksmall_4',
                           m2l=m2l_z, extinction=a_z,
                           amag_sun=SUN_Z, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellZ2 is not None:
            r, muZ2, lumZ2, errZ2 = ellZ2
            isMuZ2 = True
        else:
            muZ2, lumZ2, errZ2 = empty, empty, empty
            isMuZ2 = False
            if verbose:
                print(WAR)
                print('### Can not find the small mask SBP ' +
                      'for Z-band : %s!' % str(galID))

        """
        Z large mask
        """
        ellZ3 = sbpExtract(loc, galID, redshift,
                           'HSC-Z', prefix, rerun, 'default_msklarge_4',
                           m2l=m2l_z, extinction=a_z,
                           amag_sun=SUN_Z, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellZ3 is not None:
            r, muZ3, lumZ3, errZ3 = ellZ3
            isMuZ3 = True
        else:
            muZ3, lumZ3, errZ3 = empty, empty, empty
            isMuZ3 = False
            if verbose:
                print(WAR)
                print('### Can not find the large mask SBP ' +
                      'for Z-band : %s!' % str(galID))

        """
        Y default
        """
        ellY1 = sbpExtract(loc, galID, redshift,
                           'HSC-Y', prefix, rerun, 'default_4',
                           m2l=m2l_y, extinction=a_y,
                           amag_sun=SUN_Y, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellY1 is not None:
            r, muY1, lumY1, errY1 = ellY1
            isMuY1 = True
        else:
            muY1, lumY1, errY1 = empty, empty, empty
            isMuY1 = False
            if verbose:
                print(WAR)
                print('### Can not find the default SBP ' +
                      'for Y-band : %s!' % str(galID))

        """
        Y PSF
        """
        psfY = sbpExtract(loc, galID, redshift,
                          'HSC-Y', prefix, rerun, '3',
                          m2l=m2l_y, extinction=0.0,
                          amag_sun=SUN_Y, verbose=verbose,
                          interp=interp, psf=True)
        if psfY is not None:
            r, psfMuY, temp1, temp2 = psfY
            isPsfY = True
        else:
            psfMuY = empty
            isPsfY = False
            if verbose:
                print(WAR)
                print('### Can not find the PSF SBP ' +
                      'for Y-band : %s!' % str(galID))

        """
        Y small mask
        """
        ellY2 = sbpExtract(loc, galID, redshift,
                           'HSC-Y', prefix, rerun, 'default_msksmall_4',
                           m2l=m2l_y, extinction=a_y,
                           amag_sun=SUN_Y, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellY2 is not None:
            r, muY2, lumY2, errY2 = ellY2
            isMuY2 = True
        else:
            muY2, lumY2, errY2 = empty, empty, empty
            isMuY2 = False
            if verbose:
                print(WAR)
                print('### Can not find the small mask SBP ' +
                      'for Y-band : %s!' % str(galID))

        """
        Y large mask
        """
        ellY3 = sbpExtract(loc, galID, redshift,
                           'HSC-Y', prefix, rerun, 'default_msklarge_4',
                           m2l=m2l_y, extinction=a_y,
                           amag_sun=SUN_Y, verbose=verbose,
                           interp=interp, imgSub=imgSub)
        if ellY3 is not None:
            r, muY3, lumY3, errY3 = ellY3
            isMuY3 = True
        else:
            muY3, lumY3, errY3 = empty, empty, empty
            isMuY3 = False
            if verbose:
                print(WAR)
                print('### Can not find the large mask SBP ' +
                      'for Y-band : %s!' % str(galID))

        """ Save the summary table """
        try:
            sbpTable = Table([rad, muI1, lumI1, errI1,
                              ell, ellErr, pa, paErr,
                              muI2, lumI2, errI2,
                              muI3, lumI3, errI3,
                              muI4, lumI4, errI4,
                              muI5, lumI5, errI5,
                              muI6, lumI6, errI6,
                              muG1, lumG1, errG1,
                              muG2, lumG2, errG2,
                              muG3, lumG3, errG3,
                              muR1, lumR1, errR1,
                              muR2, lumR2, errR2,
                              muR3, lumR3, errR3,
                              muZ1, lumZ1, errZ1,
                              muZ2, lumZ2, errZ2,
                              muZ3, lumZ3, errZ3,
                              muY1, lumY1, errY1,
                              muY2, lumY2, errY2,
                              muY3, lumY3, errY3,
                              psfMuG, psfMuR, psfMuI,
                              psfMuZ, psfMuY
                              ],
                             names=('rKpc', 'muI1', 'lumI1', 'errI1',
                                    'ell', 'ellErr', 'pa', 'paErr',
                                    'muI2', 'lumI2', 'errI2',
                                    'muI3', 'lumI3', 'errI3',
                                    'muI4', 'lumI4', 'errI4',
                                    'muI5', 'lumI5', 'errI5',
                                    'muI6', 'lumI6', 'errI6',
                                    'muG1', 'lumG1', 'errG1',
                                    'muG2', 'lumG2', 'errG2',
                                    'muG3', 'lumG3', 'errG3',
                                    'muR1', 'lumR1', 'errR1',
                                    'muR2', 'lumR2', 'errR2',
                                    'muR3', 'lumR3', 'errR3',
                                    'muZ1', 'lumZ1', 'errZ1',
                                    'muZ2', 'lumZ2', 'errZ2',
                                    'muZ3', 'lumZ3', 'errZ3',
                                    'muY1', 'lumY1', 'errY1',
                                    'muY2', 'lumY2', 'errY2',
                                    'muY3', 'lumY3', 'errY3',
                                    'psfG', 'psfR', 'psfI', 'psfZ', 'psfY'),
                             meta={'LOCATION': loc,
                                   'GALID': galID,
                                   'REDSHIFT': redshift,
                                   'PREFIX': prefix,
                                   'A_G': a_g,
                                   'A_R': a_r,
                                   'A_I': a_i,
                                   'A_Z': a_z,
                                   'A_Y': a_y,
                                   'CONTAM': contaminated,
                                   'GEOM': isGeom,
                                   'MU_I2': isMuI2,
                                   'MU_I3': isMuI3,
                                   'MU_I4': isMuI4,
                                   'MU_I5': isMuI5,
                                   'MU_I6': isMuI6,
                                   'MU_G1': isMuG1,
                                   'MU_G2': isMuG2,
                                   'MU_G3': isMuG3,
                                   'MU_R1': isMuR1,
                                   'MU_R2': isMuR2,
                                   'MU_R3': isMuR3,
                                   'MU_Z1': isMuZ1,
                                   'MU_Z2': isMuZ2,
                                   'MU_Z3': isMuZ3,
                                   'MU_Y1': isMuY1,
                                   'MU_Y2': isMuY2,
                                   'MU_Y3': isMuY3,
                                   'PSF_I': isPsfI,
                                   'PSF_G': isPsfG,
                                   'PSF_R': isPsfR,
                                   'PSF_Z': isPsfZ,
                                   'PSF_Y': isPsfY
                                   })
            if save:
                sbpTable.write(sumTable, format='fits',
                               overwrite=True)
            return sbpTable
        except ValueError:
            print(WAR)
            print("## Inconsistent SBPs for %s" % str(galID))
            return None
    else:
        rad, muI1, lumI1, errI1 = None, None, None, None
        if verbose:
            print(WAR)
            warnings.warn('### Model is not available at %s for %s' %
                          (loc, str(galID)))
        return None


def correctProf(ellProf, redshift, extinction=0.0, zp=27.0,
                amag_sun=None, dimming=True, corCurve=True,
                verbose=False, m2l=None, z0=0.1, interp=True,
                blahblah=False):
    """
    Photometric correction of the Ellipse profile.

    Parameters:
    """
    """ Get physical pixel scale and distant module """
    scale = hUtil.cosmoScale(redshift)
    distmod = hUtil.cosmoDistMod(redshift)
    if dimming:
        dim = getDimming(redshift, z0)
    else:
        dim = 0.0
    if blahblah:
        print("### REDSHIFT  : %7.4f" % redshift)
        print("### PIX SCALE : %7.4f kpc/arcsec" % scale)
        print("### DIST_MOD  : %7.4f mag" % distmod)
        print("### DIMMING   : %7.4f mag" % dim)

    """ Convert unit of major axis radius to Kpc """
    sma_kpc = ellProf['sma_asec'] * scale

    if not corCurve:
        abs_mag = (-2.5 * np.log10(ellProf['growth_ori']) +
                   zp - extinction - distmod)
    else:
        abs_mag = (-2.5 * np.log10(ellProf['growth_cor']) +
                   zp - extinction - distmod)

    """ Extinction and/or Dimming corrected SBP """
    if corCurve:
        try:
            sbp_use = (ellProf['sbp_cor'] - extinction - dim)
        except KeyError:
            sbp_use = (ellProf['sbp'] - extinction - dim)
    else:
        try:
            sbp_use = (ellProf['sbp_ori'] - extinction - dim)
        except KeyError:
            sbp_use = (ellProf['sbp'] - extinction - dim)

    sbp_cor_upp = (ellProf['sbp_upp'] - extinction - dim)
    sbp_err = (sbp_cor_upp - sbp_use)

    """
    If absolute magnitude of the Sun is provided,
    Convert the SBP into physical unit of (L_sun/kpc**2)
    """
    if amag_sun is not None:
        abs_sbp = ((amag_sun + 21.572 - sbp_use) / 2.5 + 6.0)
        abs_sbp_low = ((amag_sun + 21.572 - sbp_cor_upp) / 2.5 + 6.0)
        err_sbp = (abs_sbp - abs_sbp_low)
        """
        Convert the absolute magnitude into log(L*/L_sun)
        """
        abs_mag = ((amag_sun - abs_mag) / 2.5)
        """
        If M/L ratio is provided, furthur convert it into
        surface stellar mass density profile
        """
        if m2l is not None:
            abs_sbp += np.log10(m2l)
            abs_mag += np.log10(m2l)
    else:
        abs_sbp = sbp_use
        err_sbp = sbp_err

    if not interp:
        return sma_kpc, abs_sbp, abs_mag, err_sbp
    else:
        rsma_kpc = (sma_kpc ** 0.25)
        abs_sbp_interp = interpSbp(rsma_kpc, abs_sbp,
                                   radCommon=RSMA_COMMON,
                                   kind='slinear')
        abs_mag_interp = interpSbp(rsma_kpc, abs_mag,
                                   radCommon=RSMA_COMMON,
                                   kind='slinear')
        err_sbp_interp = interpSbp(rsma_kpc, err_sbp,
                                   radCommon=RSMA_COMMON,
                                   kind='slinear')
        smaKpc_common = (RSMA_COMMON ** 4.0)
        return smaKpc_common, abs_sbp_interp, abs_mag_interp, err_sbp_interp


def coaddCutoutSbpSummary(inCat, prefix, root=None, idCol='ID', zCol='Z',
                          logmCol='MSTAR', logmErrCol='MSTAR_ERR',
                          raCol='RA', decCol='DEC',
                          agCol='a_g', arCol='a_r', aiCol='a_i',
                          azCol='a_z', ayCol='a_y', rerun='default',
                          gmagCol='gmag_cmodel', rmagCol='rmag_cmodel',
                          imagCol='imag_cmodel', zmagCol='zmag_cmodel',
                          ymagCol='zmag_cmodel', refFilter='HSC-I',
                          verbose=False, interp=True, sbpRef='lumI1',
                          sumFolder='sbp_sum', suffix=None, sample=None,
                          plot=True, imgSub=False):
    """
    Summarize the Ellipse results.

    Parameters:
        incat      :   Start with an input catalog
        root       :   Location of the data
    """
    if not os.path.isfile(inCat):
        raise Exception("## Can not find the input catalog : %s !" % inCat)

    """Name of the output catalog."""
    if suffix is None:
        strTemp = '_sbpsum'
    else:
        strTemp = '_sbpsum_' + str(suffix).strip()
    if imgSub:
        strTemp += '_imgsub'
    else:
        strTemp += '_img'
    outCat = inCat.replace('.fits', strTemp + '.fits')
    outPkl = inCat.replace('.fits', strTemp + '.pkl')

    """Read in the catalog"""
    inTab = Table.read(inCat, format='fits')
    colNames = inTab.colnames
    outTab = copy.deepcopy(inTab)

    """Check the necessary columns"""
    if (idCol not in colNames) or (zCol not in colNames):
        raise Exception("## Can not find columns for ID and Redshfit!")

    """Check the Galactic extinction correction"""
    # HSC-G
    if agCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-G band!")
        else:
            if verbose:
                print(SEP)
                print("## Estimate the Galactic Extinction in HSC-G band")
            aG = getExtinction(outTab[raCol], outTab[decCol], a_lambda=A_G)
            outTab.add_column(Column(name='a_g', data=aG))
            agCol = 'a_g'

    # HSC-R
    if arCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-R band!")
        else:
            if verbose:
                print(SEP)
                print("## Estimate the Galactic Extinction in HSC-R band")
            aR = getExtinction(outTab[raCol], outTab[decCol], a_lambda=A_R)
            outTab.add_column(Column(name='a_r', data=aR))
            arCol = 'a_r'

    # HSC-I
    if aiCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-I band!")
        else:
            if verbose:
                print(SEP)
                print("## Estimate the Galactic Extinction in HSC-I band")
            aI = getExtinction(outTab[raCol], outTab[decCol], a_lambda=A_I)
            outTab.add_column(Column(name='a_i', data=aI))
            aiCol = 'a_i'

    # HSC-Z
    if azCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-Z band!")
        else:
            if verbose:
                print(SEP)
                print("## Estimate the Galactic Extinction in HSC-Z band")
            aZ = getExtinction(outTab[raCol], outTab[decCol], a_lambda=A_Z)
            outTab.add_column(Column(name='a_z', data=aZ))
            azCol = 'a_z'

    # HSC-Y
    if ayCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-Y band!")
        else:
            if verbose:
                print(SEP)
                print("## Estimate the Galactic Extinction in HSC-Y band")
            aY = getExtinction(outTab[raCol], outTab[decCol], a_lambda=A_Y)
            outTab.add_column(Column(name='a_y', data=aY))
            ayCol = 'a_y'

    """Estimate the 'logM2L'"""
    # HSC-G
    if (logmCol not in colNames) or (gmagCol not in colNames):
        if verbose:
            print(WAR)
            print("## Can not estimate log(M/L) in HSC-G band")
        outTab.add_column(Column(name='logm2l_g', data=(outTab[zCol] * 0.0)))
    else:
        logm2lG = outTab[logmCol] - getLuminosity(outTab[gmagCol],
                                                  outTab[zCol], amag_sun=SUN_G,
                                                  extinction=outTab[agCol])
        outTab.add_column(Column(name='logm2l_g', data=logm2lG))

    # HSC-R
    if (logmCol not in colNames) or (rmagCol not in colNames):
        if verbose:
            print(WAR)
            print("## Can not estimate log(M/L) in HSC-R band")
        outTab.add_column(Column(name='logm2l_r', data=(outTab[zCol] * 0.0)))
    else:
        logm2lR = outTab[logmCol] - getLuminosity(outTab[rmagCol],
                                                  outTab[zCol], amag_sun=SUN_R,
                                                  extinction=outTab[arCol])
        outTab.add_column(Column(name='logm2l_r', data=logm2lR))

    # HSC-I
    if (logmCol not in colNames) or (imagCol not in colNames):
        if verbose:
            print(WAR)
            print("## Can not estimate log(M/L) in HSC-I band")
        outTab.add_column(Column(name='logm2l_i', data=(outTab[zCol] * 0.0)))
    else:
        logm2lI = outTab[logmCol] - getLuminosity(outTab[imagCol],
                                                  outTab[zCol], amag_sun=SUN_I,
                                                  extinction=outTab[aiCol])
        outTab.add_column(Column(name='logm2l_i', data=logm2lI))
    # HSC-Z
    if (logmCol not in colNames) or (zmagCol not in colNames):
        if verbose:
            print(WAR)
            print("## Can not estimate log(M/L) in HSC-Z band")
        outTab.add_column(Column(name='logm2l_z', data=(outTab[zCol] * 0.0)))
    else:
        logm2lZ = outTab[logmCol] - getLuminosity(outTab[zmagCol],
                                                  outTab[zCol], amag_sun=SUN_Z,
                                                  extinction=outTab[azCol])
        outTab.add_column(Column(name='logm2l_z', data=logm2lZ))

    # HSC-Y
    if (logmCol not in colNames) or (ymagCol not in colNames):
        if verbose:
            print(WAR)
            print("## Can not estimate log(M/L) in HSC-Y band")
        outTab.add_column(Column(name='logm2l_y', data=(outTab[zCol] * 0.0)))
    else:
        logm2lY = outTab[logmCol] - getLuminosity(outTab[ymagCol],
                                                  outTab[zCol], amag_sun=SUN_Y,
                                                  extinction=outTab[ayCol])
        outTab.add_column(Column(name='logm2l_y', data=logm2lY))

    """Add columns to the output table"""
    colTemp = (np.asarray(outTab[zCol]) * 0.0 - 9999.0)
    col1 = Column(name='lum_max', data=colTemp)
    col2 = Column(name='lum_150', data=colTemp)
    col3 = Column(name='lum_120', data=colTemp)
    col4 = Column(name='lum_100', data=colTemp)
    col5 = Column(name='lum_75', data=colTemp)
    col6 = Column(name='lum_50', data=colTemp)
    col7 = Column(name='lum_25', data=colTemp)
    col8 = Column(name='lum_10', data=colTemp)
    col9 = Column(name='lum_5', data=colTemp)
    col10 = Column(name='lum_15', data=colTemp)
    col11 = Column(name='lum_30', data=colTemp)
    col12 = Column(name='lum_40', data=colTemp)
    col13 = Column(name='lum_60', data=colTemp)
    col14 = Column(name='r20_max', data=colTemp)
    col15 = Column(name='r50_max', data=colTemp)
    col16 = Column(name='r80_max', data=colTemp)
    col17 = Column(name='r90_max', data=colTemp)
    col18 = Column(name='r20_120', data=colTemp)
    col19 = Column(name='r50_120', data=colTemp)
    col20 = Column(name='r80_120', data=colTemp)
    col21 = Column(name='r90_120', data=colTemp)
    col22 = Column(name='r20_100', data=colTemp)
    col23 = Column(name='r50_100', data=colTemp)
    col24 = Column(name='r80_100', data=colTemp)
    col25 = Column(name='r90_100', data=colTemp)
    col26 = Column(name='c82_max', data=colTemp)
    col27 = Column(name='c82_120', data=colTemp)
    col28 = Column(name='c82_100', data=colTemp)
    outTab.add_columns([col1, col2, col3, col4, col5, col6, col7, col8,
                        col9, col10, col11, col12, col13, col14, col15,
                        col16, col17, col18, col19, col20, col21, col22,
                        col23, col24, col25, col26, col27, col28])

    """Start a ProgressBar"""
    sbpSum = []
    sbpList = []
    with ProgressBar(len(outTab)) as bar:
        if verbose:
            print(SEP)
            print("## Dealing with %d galaxies" % len(outTab))
            print(SEP)
        """Go through the catalog to search for data"""
        for (ii, galaxy) in enumerate(outTab):
            """ID of the galaxy"""
            galID = galaxy[idCol]
            galStr = str(galID).strip()
            """Redshift"""
            galZ = galaxy[zCol]
            """Data Dir"""
            if root is None:
                loc = '.'
            else:
                loc = root
            """Folder for summary table"""
            sumDir = os.path.join(loc, sumFolder)
            if not os.path.isdir(sumDir):
                os.mkdir(sumDir)

            """Summary table"""
            if sample is None:
                sumCat = galStr + '_sbpsum'
            else:
                sumCat = str(sample).strip() + '_' + galStr + '_sbpsum'
            if imgSub:
                sumCat += '_imgsub'
            else:
                sumCat += '_img'
            if suffix is None:
                sumCat = sumCat + '.fits'
            else:
                sumCat = sumCat + '_' + str(suffix).strip() + '.fits'

            sumCat = os.path.join(sumDir, sumCat)
            if plot:
                sumPng = sumCat.replace('.fits', '.png')

            """
            Get the collection of SBP results

            Right now, don't automatically convert into mass profile
            """
            galTab = sbpCollect(loc, prefix, galStr, galZ,
                                a_g=galaxy[agCol],
                                a_r=galaxy[arCol],
                                a_i=galaxy[aiCol],
                                a_z=galaxy[azCol],
                                a_y=galaxy[ayCol],
                                verbose=verbose, save=False,
                                sumFolder=sumFolder,
                                sample=sample,
                                suffix=suffix,
                                imgSub=imgSub)

            if galTab is not None:
                """Radius KPc"""
                rKpc = galTab['rKpc']
                """Maximum luminosity from certain model"""
                lumRef = galTab[sbpRef]
                """Get the 1-D interpolation function"""
                radInterp = interp1d(rKpc, lumRef)
                """Maximum intergrated 1-d luminosity"""
                lumMax = np.nanmax(lumRef).astype(np.float32)
                if not np.isfinite(lumMax):
                    lumMax = radInterp(150.0)
                """Out to 150 Kpc"""
                lum150 = np.nanmax(lumRef[rKpc <= 150.0]).astype(np.float32)
                lum150i = radInterp(150.0)
                lum150 = lum150 if (lum150 >= lum150i) else lum150i
                """Out to 120 Kpc"""
                lum120 = np.nanmax(lumRef[rKpc <= 120.0]).astype(np.float32)
                lum120i = radInterp(120.0)
                lum120 = lum120 if (lum120 >= lum120i) else lum120i
                """Out to 100 Kpc"""
                lum100 = np.nanmax(lumRef[rKpc <= 100.0]).astype(np.float32)
                lum100i = radInterp(100.0)
                lum100 = lum100 if (lum100 >= lum100i) else lum100i
                """Out to 75 Kpc"""
                lum75 = np.nanmax(lumRef[rKpc <= 75.0]).astype(np.float32)
                lum75i = radInterp(75.0)
                lum75 = lum75 if (lum75 >= lum75i) else lum75i
                """Out to 60 Kpc"""
                lum60 = np.nanmax(lumRef[rKpc <= 60.0]).astype(np.float32)
                lum60i = radInterp(60.0)
                lum60 = lum60 if (lum60 >= lum60i) else lum60i
                """Out to 50 Kpc"""
                lum50 = np.nanmax(lumRef[rKpc <= 50.0]).astype(np.float32)
                lum50i = radInterp(50.0)
                lum50 = lum50 if (lum50 >= lum50i) else lum50i
                """Out to 40 Kpc"""
                lum40 = np.nanmax(lumRef[rKpc <= 40.0]).astype(np.float32)
                lum40i = radInterp(40.0)
                lum40 = lum40 if (lum40 >= lum40i) else lum40i
                """Out to 30 Kpc"""
                lum30 = np.nanmax(lumRef[rKpc <= 30.0]).astype(np.float32)
                lum30i = radInterp(30.0)
                lum30 = lum30 if (lum30 >= lum30i) else lum30i
                """Out to 25 Kpc"""
                lum25 = np.nanmax(lumRef[rKpc <= 25.0]).astype(np.float32)
                lum25i = radInterp(25.0)
                lum25 = lum25 if (lum25 >= lum25i) else lum25i
                """Out to 15 Kpc"""
                lum15 = np.nanmax(lumRef[rKpc <= 15.0]).astype(np.float32)
                lum15i = radInterp(15.0)
                lum15 = lum15 if (lum15 >= lum15i) else lum15i
                """Out to 10 Kpc"""
                lum10 = np.nanmax(lumRef[rKpc <= 10.0]).astype(np.float32)
                lum10i = radInterp(10.0)
                lum10 = lum10 if (lum10 >= lum10i) else lum10i
                """Out to 5 Kpc"""
                lum5 = np.nanmax(lumRef[rKpc <= 5.0]).astype(np.float32)
                lum5i = radInterp(5.0)
                lum5 = lum5 if (lum5 >= lum5i) else lum5i

                if not np.isfinite(lum120):
                    if verbose:
                        print(WAR)
                        print("## Problematic SBP for %s !" % galStr)
                    lumMax = -9999.0
                    lum150 = -9999.0
                    lum120 = -9999.0
                    lum100 = -9999.0
                    lum75 = -9999.0
                    lum60 = -9999.0
                    lum50 = -9999.0
                    lum40 = -9999.0
                    lum30 = -9999.0
                    lum25 = -9999.0
                    lum15 = -9999.0
                    lum10 = -9999.0
                    lum5 = -9999.0

                galTab.meta['LUM_MAX'] = float(lumMax)
                galTab.meta['LUM_150'] = float(lum150)
                galTab.meta['LUM_120'] = float(lum120)
                galTab.meta['LUM_100'] = float(lum100)
                galTab.meta['LUM_75'] = float(lum75)
                galTab.meta['LUM_60'] = float(lum60)
                galTab.meta['LUM_50'] = float(lum50)
                galTab.meta['LUM_40'] = float(lum40)
                galTab.meta['LUM_30'] = float(lum30)
                galTab.meta['LUM_25'] = float(lum25)
                galTab.meta['LUM_15'] = float(lum15)
                galTab.meta['LUM_10'] = float(lum10)
                galTab.meta['LUM_5'] = float(lum5)

                """M2L"""
                galTab.meta['LOGM2L_G'] = galaxy['logm2l_g']
                galTab.meta['LOGM2L_R'] = galaxy['logm2l_r']
                galTab.meta['LOGM2L_I'] = galaxy['logm2l_i']
                galTab.meta['LOGM2L_Z'] = galaxy['logm2l_z']
                galTab.meta['LOGM2L_Y'] = galaxy['logm2l_y']

                """Save the result of the individual galaxy"""
                galTab.write(sumCat, format='fits', overwrite=True)

                """Update the sample summary table"""
                outTab['lum_max'][ii] = lumMax
                outTab['lum_150'][ii] = lum150
                outTab['lum_120'][ii] = lum120
                outTab['lum_100'][ii] = lum100
                outTab['lum_75'][ii] = lum75
                outTab['lum_60'][ii] = lum60
                outTab['lum_50'][ii] = lum50
                outTab['lum_40'][ii] = lum40
                outTab['lum_30'][ii] = lum30
                outTab['lum_25'][ii] = lum25
                outTab['lum_15'][ii] = lum15
                outTab['lum_10'][ii] = lum10
                outTab['lum_5'][ii] = lum5

                """Get the R20, R50, R80 and R90"""
                if np.isfinite(lumMax):
                    try:
                        fracMax = (10.0 ** lumRef) / (10.0 ** lumMax)
                        fracInterp1 = interp1d(fracMax, rKpc)
                        outTab['r20_max'][ii] = fracInterp1(0.20)
                        outTab['r50_max'][ii] = fracInterp1(0.50)
                        outTab['r80_max'][ii] = fracInterp1(0.80)
                        outTab['r90_max'][ii] = fracInterp1(0.80)
                        outTab['c82_max'][ii] = (outTab['r80_max'][ii] /
                                                 outTab['r20_max'][ii])
                    except Exception:
                        outTab['r20_max'][ii] = -9999.0
                        outTab['r50_max'][ii] = -9999.0
                        outTab['r80_max'][ii] = -9999.0
                        outTab['r90_max'][ii] = -9999.0
                        outTab['c82_max'][ii] = -9999.0
                else:
                    outTab['r20_max'][ii] = -9999.0
                    outTab['r50_max'][ii] = -9999.0
                    outTab['r80_max'][ii] = -9999.0
                    outTab['r90_max'][ii] = -9999.0
                    outTab['c82_max'][ii] = -9999.0

                if np.isfinite(lum120):
                    try:
                        frac120 = (10.0 ** lumRef) / (10.0 ** lum120)
                        fracInterp2 = interp1d(frac120, rKpc)
                        outTab['r20_120'][ii] = fracInterp2(0.20)
                        outTab['r50_120'][ii] = fracInterp2(0.50)
                        outTab['r80_120'][ii] = fracInterp2(0.80)
                        outTab['r90_120'][ii] = fracInterp2(0.80)
                        outTab['c82_120'][ii] = (outTab['r80_120'][ii] /
                                                 outTab['r20_120'][ii])
                    except Exception:
                        outTab['r20_120'][ii] = -9999.0
                        outTab['r50_120'][ii] = -9999.0
                        outTab['r80_120'][ii] = -9999.0
                        outTab['r90_120'][ii] = -9999.0
                        outTab['c82_120'][ii] = -9999.0
                else:
                    outTab['r20_120'][ii] = -9999.0
                    outTab['r50_120'][ii] = -9999.0
                    outTab['r80_120'][ii] = -9999.0
                    outTab['r90_120'][ii] = -9999.0
                    outTab['c82_120'][ii] = -9999.0

                if np.isfinite(lum100):
                    try:
                        frac100 = (10.0 ** lumRef) / (10.0 ** lum100)
                        fracInterp3 = interp1d(frac100, rKpc)
                        outTab['r20_100'][ii] = fracInterp3(0.20)
                        outTab['r50_100'][ii] = fracInterp3(0.50)
                        outTab['r80_100'][ii] = fracInterp3(0.80)
                        outTab['r90_100'][ii] = fracInterp3(0.80)
                        outTab['c82_100'][ii] = (outTab['r80_100'][ii] /
                                                 outTab['r20_100'][ii])
                    except Exception:
                        outTab['r20_100'][ii] = -9999.0
                        outTab['r50_100'][ii] = -9999.0
                        outTab['r80_100'][ii] = -9999.0
                        outTab['r90_100'][ii] = -9999.0
                        outTab['c82_100'][ii] = -9999.0
                else:
                    outTab['r20_100'][ii] = -9999.0
                    outTab['r50_100'][ii] = -9999.0
                    outTab['r80_100'][ii] = -9999.0
                    outTab['r90_100'][ii] = -9999.0
                    outTab['c82_100'][ii] = -9999.0

                """Extra meta information"""
                galTab.meta['R20_MAX'] = outTab['r20_max'][ii]
                galTab.meta['R50_MAX'] = outTab['r50_max'][ii]
                galTab.meta['R80_MAX'] = outTab['r80_max'][ii]
                galTab.meta['R90_MAX'] = outTab['r90_max'][ii]
                galTab.meta['C82_MAX'] = outTab['c82_max'][ii]

                galTab.meta['R20_120'] = outTab['r20_120'][ii]
                galTab.meta['R50_120'] = outTab['r50_120'][ii]
                galTab.meta['R80_120'] = outTab['r80_120'][ii]
                galTab.meta['R90_120'] = outTab['r90_120'][ii]
                galTab.meta['C82_120'] = outTab['c82_120'][ii]

                galTab.meta['R20_100'] = outTab['r20_100'][ii]
                galTab.meta['R50_100'] = outTab['r50_100'][ii]
                galTab.meta['R80_100'] = outTab['r80_100'][ii]
                galTab.meta['R90_100'] = outTab['r90_100'][ii]
                galTab.meta['C82_100'] = outTab['c82_100'][ii]

                """"""
                sbpSum.append(galTab)
                sbpList.append(sumCat)

                if plot:
                    sbpCompare(galTab, sumPng)
            else:
                """"""
                sbpSum.append(None)
                sbpList.append('None')
                """"""
                warnings.warn('### NO USEFUL DATA FOR %s' % galStr)

            """Update the progress bar"""
            bar.update()

        """Save a Pickle file of the results"""
        hUtil.saveToPickle(sbpSum, outPkl)

    """Save the output catalog"""
    if verbose:
        print("# Summary table saved to : %s" % outCat)
    outTab.add_column(Column(np.asarray(sbpList), name='sum_tab'))
    outTab.write(outCat, format='fits', overwrite=True)

    return outCat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("incat", help="Input catalog")
    parser.add_argument("prefix", help="Prefix for the dataset")
    parser.add_argument('-r', '--root', dest='root',
                        help='Path to the results',
                        default=None)
    parser.add_argument('--id', '--idCol', dest='idCol',
                        help='Column for galaxy ID',
                        default='ID')
    parser.add_argument('--z', '--zCol', dest='zCol',
                        help='Column for redshift',
                        default='Z')
    parser.add_argument('--ra', '--raCol', dest='raCol',
                        help='Column for RA',
                        default='RA')
    parser.add_argument('--dec', '--decCol', dest='decCol',
                        help='Column for DEC',
                        default='DEC')
    parser.add_argument('--logm', '--logmCol', dest='logmCol',
                        help='Column for logMstar',
                        default='MSTAR')
    parser.add_argument('--logmerr', '--logmErrCol', dest='logmErrCol',
                        help='Column for logMstarErr',
                        default='MSTAR_ERR')
    parser.add_argument('--rerun', dest='rerun',
                        help='Name of the rerun',
                        default='default')
    parser.add_argument('--sumFolder', dest='sumFolder',
                        help='Name of the folder for the summary',
                        default='sbp_sum')
    parser.add_argument('--sbpRef', dest='sbpRef',
                        help='Name of the reference SBP',
                        default='lumI1')
    parser.add_argument('--suffix', dest='suffix',
                        help='Suffix of the summary file',
                        default=None)
    parser.add_argument('--sample', dest='sample',
                        help='Name of the sample',
                        default=None)
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action="store_true",
                        default=False)
    parser.add_argument('-p', '--plot', dest='plot',
                        action="store_true",
                        default=False)
    parser.add_argument('--imgSub', dest='imgSub',
                        action="store_true",
                        default=False)

    args = parser.parse_args()

    coaddCutoutSbpSummary(args.incat, args.prefix,
                          root=args.root,
                          idCol=args.idCol,
                          zCol=args.zCol,
                          logmCol=args.logmCol,
                          logmErrCol=args.logmErrCol,
                          raCol=args.raCol,
                          decCol=args.decCol,
                          rerun=args.rerun,
                          verbose=args.verbose,
                          sumFolder=args.sumFolder,
                          sbpRef=args.sbpRef,
                          suffix=args.suffix,
                          sample=args.sample,
                          plot=args.plot,
                          imgSub=args.imgSub)
