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
# from scipy.interpolate import interp1d

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
RSMA_COMMON = np.arange(0.2, 4.0, 0.1)
EMPTY = (RSMA_COMMON * np.nan)
"""
For output
"""
COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


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


def getDimming(z1, z0=0.1):
    """
    Get the surface brightness dimming effect.

    Parameters:
        z1: Observed redshift
        z0: Reference redshift
            Default = 0.1
    """
    return (3.0 * np.log10((1.0 + z1) / (1.0 + z0)))


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
                    filter='HSC-I', rerun='default', verbose=False):
    """
    Find and load the Ellipse output.

    Parameters:
    """
    galid = str(galid).strip()
    location = os.path.join(base, galid, filter, rerun)
    if psf:
        ellType = 'psf'
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


def sbpExtract(loc, galID, redshift, filter,
               prefix, rerun, model,
               zp=27.0, extinction=0.0,
               amag_sun=None, m2l=None,
               origin=False, verbose=False):
    """
    Return important SBP information.

    Parameters:
    """
    prof = getEllipProfile(galID, loc, prefix, model,
                           filter=filter, rerun=rerun,
                           verbose=verbose)
    if prof is not None:
        ell = correctProf(prof, redshift,
                          extinction=extinction,
                          zp=zp, amag_sun=amag_sun,
                          dimming=True, corCurve=True,
                          verbose=verbose, m2l=m2l)
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
               m2l_z=None, m2l_y=None,
               verbose=False, save=True):
    """
    Collect profiles from the cutout folder.

    This is a quick work-around, eventually should us a Class for dataset

    This works for the redBCG and nonBCG datasets 2015-12-09
    And, can not use for any profile with suffix
    """
    """Location and Table name"""
    sumDir = os.path.join(loc, 'sum')
    if not os.path.isdir(sumDir):
        os.mkdir(sumDir)
    """Name of the summary table of each galaxy"""
    if suffix is None:
        sumTab = str(galID) + '_sbp_sum.fits'
        suffix = ''
    else:
        sumTab = str(galID) + '_' + suffix + '_sbp_sum.fits'

    if not os.path.exists(sumDir):
        os.mkdir(sumDir)
    sumTable = os.path.join(sumDir, sumTab)

    """ The basic reference model """
    refEllI = sbpExtract(loc, galID, redshift, 'HSC-I',
                         prefix, rerun, 'default_3',
                         extinction=a_i, m2l=m2l_i,
                         amag_sun=SUN_I, verbose=verbose)
    if refEllI is not None:
        """ Reference profile in I-band """
        rad, muI1, lumI1, errI1 = refEllI
        """ Create a NaN array """
        empty = copy.deepcopy(rad)
        empty[:] = np.nan

        """ I largeR1 """
        ellI2 = sbpExtract(loc, galID, redshift,
                           'HSC-I', prefix, rerun, 'multi1_4',
                           m2l=m2l_i, extinction=a_i,
                           amag_sun=SUN_I, verbose=verbose)
        if ellI2 is not None:
            r, muI2, lumI2, errI2 = ellI2
        else:
            muI2, lumI2, errI2 = empty, empty, empty
            print(WAR)
            print('### Can not find the small mask SBP ' +
                  'for I-band : %s!' % str(galID))

        """ I smallR1 """
        ellI3 = sbpExtract(loc, galID, redshift,
                           'HSC-I', prefix, rerun, 'multi2_4',
                           m2l=m2l_i, extinction=a_i,
                           amag_sun=SUN_I, verbose=verbose)
        if ellI3 is not None:
            r, muI3, lumI3, errI3 = ellI3
        else:
            muI3, lumI3, errI3 = empty, empty, empty
            print(WAR)
            print('### Can not find the large mask SBP ' +
                  'for I-band : %s!' % str(galID))

        """ G default """
        ellG1 = sbpExtract(loc, galID, redshift,
                           'HSC-G', prefix, rerun, 'default_4',
                           m2l=m2l_g, extinction=a_g,
                           amag_sun=SUN_G, verbose=verbose)
        if ellG1 is not None:
            r, muG1, lumG1, errG1 = ellG1
        else:
            muG1, lumG1, errG1 = empty, empty, empty
            print(WAR)
            print('### Can not find the default SBP ' +
                  'for G-band : %s!' % str(galID))

        """ R default """
        ellR1 = sbpExtract(loc, galID, redshift,
                           'HSC-R', prefix, rerun, 'default_4',
                           m2l=m2l_r, extinction=a_r,
                           amag_sun=SUN_R, verbose=verbose)
        if ellR1 is not None:
            r, muR1, lumR1, errR1 = ellR1
        else:
            muR1, lumR1, errR1 = empty, empty, empty
            print(WAR)
            print('### Can not find the default SBP ' +
                  'for R-band : %s!' % str(galID))

        """ Z default """
        ellZ1 = sbpExtract(loc, galID, redshift,
                           'HSC-Z', prefix, rerun, 'default_4',
                           m2l=m2l_z, extinction=a_z,
                           amag_sun=SUN_Z, verbose=verbose)
        if ellZ1 is not None:
            r, muZ1, lumZ1, errZ1 = ellZ1
        else:
            muZ1, lumZ1, errZ1 = empty, empty, empty
            print(WAR)
            print('### Can not find the default SBP ' +
                  'for Z-band : %s!' % str(galID))

        """ Y default """
        ellY1 = sbpExtract(loc, galID, redshift,
                           'HSC-Y', prefix, rerun, 'default_4',
                           m2l=m2l_y, extinction=a_y,
                           amag_sun=SUN_Y, verbose=verbose)
        if ellY1 is not None:
            r, muY1, lumY1, errY1 = ellY1
        else:
            muY1, lumY1, errY1 = empty, empty, empty
            print(WAR)
            print('### Can not find the default SBP ' +
                  'for Y-band : %s!' % str(galID))

        """ Save the summary table """
        try:
            sbpTable = Table([rad, muI1, lumI1, errI1,
                              muI2, lumI2, errI2,
                              muI3, lumI3, errI3,
                              muG1, lumG1, errG1,
                              muR1, lumR1, errR1,
                              muZ1, lumZ1, errZ1,
                              muY1, lumY1, errY1,
                              ],
                             names=('rKpc', 'muI1', 'lumI1', 'errI1',
                                    'muI2', 'lumI2', 'errI2',
                                    'muI3', 'lumI3', 'errI3',
                                    'muG1', 'lumG1', 'errG1',
                                    'muR1', 'lumR1', 'errR1',
                                    'muZ1', 'lumZ1', 'errZ1',
                                    'muY1', 'lumY1', 'errY1'),
                             meta={'LOCATION': loc,
                                   'GALID': galID,
                                   'REDSHIFT': redshift,
                                   'PREFIX': prefix,
                                   'A_G': a_g,
                                   'A_R': a_r,
                                   'A_I': a_i,
                                   'A_Z': a_z,
                                   'A_Y': a_y})
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
                verbose=False, m2l=None, z0=0.1):
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
    if verbose:
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
        abs_sbp = ((amag_sun + 21.572 - sbp_use)/2.5 + 6.0)
        abs_sbp_low = ((amag_sun + 21.572 - sbp_cor_upp)/2.5 + 6.0)
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

    return sma_kpc, abs_sbp, abs_mag, err_sbp


def coaddCutoutSbpSummary(inCat, prefix, root=None, idCol='ID', zCol='Z',
                          logmCol='MSTAR', logmErrCol='MSTAR_ERR',
                          raCol='RA', decCol='DEC',
                          agCol='a_g', arCol='a_r', aiCol='a_i',
                          azCol='a_z', ayCol='a_y', rerun='default',
                          gmagCol='gmag_cmodel', rmagCol='rmag_cmodel',
                          imagCol='imag_cmodel', zmagCol='zmag_cmodel',
                          ymagCol='zmag_cmodel', refFilter='HSC-I',
                          verbose=False):
    """
    Summarize the Ellipse results.

    Parameters:
        incat      :   Start with an input catalog
        root       :   Location of the data
    """
    if not os.path.isfile(inCat):
        raise Exception("## Can not find the input catalog : %s !" % inCat)
    """Name of the output catalog."""
    outCat = inCat.replace('.fits', '_sbpsum.fits')
    outPkl = inCat.replace('.fits', '_sbpsum.pkl')
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
            print(SEP)
            print("## Estimate the Galactic Extinction in HSC-G band")
            aG = getExtinction(inTab[raCol], inTab[decCol], a_lambda=A_G)
            inTab.add_column(Column(name='a_g', data=aG))
            agCol = 'a_g'
    # HSC-R
    if arCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-R band!")
        else:
            print(SEP)
            print("## Estimate the Galactic Extinction in HSC-R band")
            aR = getExtinction(inTab[raCol], inTab[decCol], a_lambda=A_R)
            inTab.add_column(Column(name='a_r', data=aR))
            arCol = 'a_r'
    # HSC-I
    if aiCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-I band!")
        else:
            print(SEP)
            print("## Estimate the Galactic Extinction in HSC-I band")
            aI = getExtinction(inTab[raCol], inTab[decCol], a_lambda=A_I)
            inTab.add_column(Column(name='a_i', data=aI))
            aiCol = 'a_i'
    # HSC-Z
    if azCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-Z band!")
        else:
            print(SEP)
            print("## Estimate the Galactic Extinction in HSC-Z band")
            aZ = getExtinction(inTab[raCol], inTab[decCol], a_lambda=A_Z)
            inTab.add_column(Column(name='a_z', data=aZ))
            azCol = 'a_z'
    # HSC-Y
    if ayCol not in colNames:
        if (raCol not in colNames) or (decCol not in colNames):
            raise Exception("## Can not estimate the extinction " +
                            " in HSC-Y band!")
        else:
            print(SEP)
            print("## Estimate the Galactic Extinction in HSC-Y band")
            aY = getExtinction(inTab[raCol], inTab[decCol], a_lambda=A_Y)
            inTab.add_column(Column(name='a_y', data=aY))
            ayCol = 'a_y'
    """Estimate the 'logM2L'"""
    # HSC-G
    if (logmCol not in colNames) or (gmagCol not in colNames):
        print(WAR)
        print("## Can not estimate log(M/L) in HSC-G band")
        inTab.add_column(Column(name='logm2l_g', data=(inTab[zCol] * 0.0)))
    else:
        logm2lG = inTab[logmCol] - getLuminosity(inTab[gmagCol],
                                                 inTab[zCol], amag_sun=SUN_G,
                                                 extinction=inTab[agCol])
        inTab.add_column(Column(name='logm2l_g', data=logm2lG))
    # HSC-R
    if (logmCol not in colNames) or (rmagCol not in colNames):
        print(WAR)
        print("## Can not estimate log(M/L) in HSC-R band")
        inTab.add_column(Column(name='logm2l_r', data=(inTab[zCol] * 0.0)))
    else:
        logm2lR = inTab[logmCol] - getLuminosity(inTab[rmagCol],
                                                 inTab[zCol], amag_sun=SUN_R,
                                                 extinction=inTab[arCol])
        inTab.add_column(Column(name='logm2l_r', data=logm2lR))
    # HSC-I
    if (logmCol not in colNames) or (imagCol not in colNames):
        print(WAR)
        print("## Can not estimate log(M/L) in HSC-I band")
        inTab.add_column(Column(name='logm2l_i', data=(inTab[zCol] * 0.0)))
    else:
        logm2lI = inTab[logmCol] - getLuminosity(inTab[imagCol],
                                                 inTab[zCol], amag_sun=SUN_I,
                                                 extinction=inTab[aiCol])
        inTab.add_column(Column(name='logm2l_i', data=logm2lI))
    # HSC-Z
    if (logmCol not in colNames) or (zmagCol not in colNames):
        print(WAR)
        print("## Can not estimate log(M/L) in HSC-Z band")
        inTab.add_column(Column(name='logm2l_z', data=(inTab[zCol] * 0.0)))
    else:
        logm2lZ = inTab[logmCol] - getLuminosity(inTab[zmagCol],
                                                 inTab[zCol], amag_sun=SUN_Z,
                                                 extinction=inTab[azCol])
        inTab.add_column(Column(name='logm2l_z', data=logm2lZ))
    # HSC-Y
    if (logmCol not in colNames) or (ymagCol not in colNames):
        print(WAR)
        print("## Can not estimate log(M/L) in HSC-Y band")
        inTab.add_column(Column(name='logm2l_y', data=(inTab[zCol] * 0.0)))
    else:
        logm2lY = inTab[logmCol] - getLuminosity(inTab[ymagCol],
                                                 inTab[zCol], amag_sun=SUN_Y,
                                                 extinction=inTab[ayCol])
        inTab.add_column(Column(name='logm2l_y', data=logm2lY))

    """Add columns to the output table"""
    colTemp = (np.asarray(inTab[zCol]) * 0.0 - 9999.0)
    col1 = Column(name='ilum_max', data=colTemp)
    col2 = Column(name='ilum_100', data=colTemp)
    col3 = Column(name='ilum_75', data=colTemp)
    col4 = Column(name='ilum_50', data=colTemp)
    col5 = Column(name='ilum_25', data=colTemp)
    col6 = Column(name='ilum_10', data=colTemp)
    outTab.add_columns([col1, col2, col3, col4, col5, col6])
    """Start a ProgressBar"""
    sbpSum = []
    sbpList = []
    with ProgressBar(len(inTab)) as bar:
        print(SEP)
        print("## Dealing with %d galaxies" % len(inTab))
        print(SEP)
        """Go through the catalog to search for data"""
        for (ii, galaxy) in enumerate(inTab):
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
            sumDir = os.path.join(loc, 'sbp_sum')
            if not os.path.isdir(sumDir):
                os.mkdir(sumDir)
            """Summary table"""
            sumCat = galStr + '_sbp_sum.fits'
            sumCat = os.path.join(sumDir, sumCat)
            """Get the collection of SBP results"""
            galTab = sbpCollect(loc, prefix, galStr, galZ,
                                a_g=galaxy[agCol],
                                a_r=galaxy[arCol],
                                a_i=galaxy[aiCol],
                                a_z=galaxy[azCol],
                                a_y=galaxy[ayCol],
                                verbose=verbose, save=True)
            if galTab is not None:
                """Radius KPc"""
                rKpc = galTab['rKpc']
                """Maximum i-band luminosity"""
                lumI = galTab['lumI1']
                ilumMax = np.nanmax(lumI).astype(np.float32)
                ilum100 = np.nanmax(lumI[rKpc <= 100.0]).astype(np.float32)
                ilum75 = np.nanmax(lumI[rKpc <= 75.0]).astype(np.float32)
                ilum50 = np.nanmax(lumI[rKpc <= 50.0]).astype(np.float32)
                ilum25 = np.nanmax(lumI[rKpc <= 25.0]).astype(np.float32)
                ilum10 = np.nanmax(lumI[rKpc <= 10.0]).astype(np.float32)
                if not np.isfinite(ilumMax):
                    print(WAR)
                    print("## Problematic SBP for %s !" % galStr)
                    ilumMax = -9999.0
                    ilum100 = -9999.0
                    ilum75 = -9999.0
                    ilum50 = -9999.0
                    ilum25 = -9999.0
                    ilum10 = -9999.0
                galTab.meta['ILUM_MAX'] = ilumMax
                galTab.meta['ILUM_100'] = ilum100
                galTab.meta['ILUM_75'] = ilum75
                galTab.meta['ILUM_50'] = ilum50
                galTab.meta['ILUM_25'] = ilum25
                galTab.meta['ILUM_10'] = ilum10
                """M2L"""
                galTab.meta['LOGM2L_G'] = galaxy['logm2l_g']
                galTab.meta['LOGM2L_R'] = galaxy['logm2l_r']
                galTab.meta['LOGM2L_I'] = galaxy['logm2l_i']
                galTab.meta['LOGM2L_Z'] = galaxy['logm2l_z']
                galTab.meta['LOGM2L_Y'] = galaxy['logm2l_y']
                """Save the result of the individual galaxy"""
                galTab.write(sumCat, format='fits', overwrite=True)
                """Update the sample summary table"""
                outTab['ilum_max'][ii] = ilumMax
                outTab['ilum_100'][ii] = ilum100
                outTab['ilum_75'][ii] = ilum75
                outTab['ilum_50'][ii] = ilum50
                outTab['ilum_25'][ii] = ilum25
                outTab['ilum_10'][ii] = ilum10
                """"""
                sbpSum.append(galTab)
                sbpList.append(sumCat)
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
    parser.add_argument('-v', '--verbose', dest='verbose',
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
                          verbose=args.verbose)
