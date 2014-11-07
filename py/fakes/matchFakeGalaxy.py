#!/usr/bin/env python
"""
matchFakes.py
matches fakes based on position stored in the calibrated exposure image header
"""

import lsst.daf.persistence as dafPersist
from lsst.afw.table import SourceCatalog, SchemaMapper
import lsst.afw.geom.ellipses as geomEllip
import numpy as np
import argparse
import re
import collections
import pyfits as fits

def getSizeAndShape(m):
    """
    Get the major axis radius, axis ratio, position angle from moments
    m = Moments
    """
    ixx, ixy, iyy = m.getIxx(), m.getIxy(), m.getIyy()
    # convert to an ellipse (note that theta is in radians and is not an
    # Angle object)
    ellipse = geomEllip.Axes(m)
    a, b, theta = ellipse.getA(), ellipse.getB(), ellipse.getTheta()

    pa = theta * 180.0 / np.pi

    return a, (b/a), pa


def getGalaxy(rootdir, visit, ccd, tol):
    """Get list of sources which agree in position with fake ones with tol
    """
    # Call the butler
    butler = dafPersist.Butler(rootdir)
    dataId = {'visit':visit, 'ccd':ccd}
    tol = float(tol)

    # Get the source catalog and metadata
    sources = butler.get('src', dataId)
    cal_md  = butler.get('calexp_md', dataId)

    # Get the X, Y locations of objects on the CCD
    srcX, srcY = sources.getX(), sources.getY()
    # Get the zeropoint
    zeropoint = (2.5 * np.log10(cal_md.get("FLUXMAG0")))
    # Get the parent ID
    parentID = sources.get('parent')
    # Check the star/galaxy separation
    extendClass = sources.get('classification.extendedness')

    # For Galaxies: Get these parameters
    # 1. Get the Kron flux and its error
    fluxKron, ferrKron = sources.get('flux.kron'), sources.get('flux.kron.err')
    magKron, merrKron = (zeropoint - 2.5*np.log10(fluxKron)), (2.5/np.log(10)*
                                                            (ferrKron/fluxKron))
    # 2. Get the CModel flux and its error
    fluxCmod, ferrCmod = sources.get('cmodel.flux'), sources.get('cmodel.flux.err')
    magCmod, merrCmod = (zeropoint - 2.5*np.log10(fluxCmod)), (2.5/np.log(10)*
                                                            (ferrCmod/fluxCmod))
    # 3. Get the Exponential flux and its error
    fluxExp, ferrExp = sources.get('cmodel.exp.flux'), sources.get('cmodel.exp.flux.err')
    magExp, merrExp = (zeropoint - 2.5*np.log10(fluxExp)), (2.5/np.log(10)*
                                                            (ferrExp/fluxExp))
    # 4. Get the de Vacouleurs flux and its error
    fluxDev, ferrDev = sources.get('cmodel.dev.flux'), sources.get('cmodel.dev.flux.err')
    magDev, merrDev = (zeropoint - 2.5*np.log10(fluxDev)), (2.5/np.log(10)*
                                                            (ferrDev/fluxDev))
    # 5. Get the SDSS shapes (Re, b/a, PA)
    sdssMoment = sources.get('shape.sdss')
    sdssR, sdssBa, sdssPa = getSizeAndShape(sdssMoment)
    # 6. Get the Exponential shapes (Re, b/a, PA)
    expMoment = sources.get('cmodel.exp.ellipse')
    expR, expBa, expPa = getSizeAndShape(expMoment)
    # 7. Get the de Vaucouleurs shapes (Re, b/a, PA)
    devMoment = sources.get('cmodel.dev.ellipse')
    devR, devBa, devPa = getSizeAndShape(devMoment)
    # 8. Get the fracDev
    fracDev = sources.get('cmodel.fracDev')

    # X, Y locations of the fake stars
    fakeList = collections.defaultdict(tuple)
    # Regular Expression
    # Search for keywords like FAKE12
    fakename = re.compile('FAKE([0-9]+)')
    # Go through all the keywords
    counts = 0
    for card in cal_md.names():
        # To see if the card matches the pattern
        m = fakename.match(card)
        if m is not None:
            # Get the X,Y location for fake object
            x,y    = map(float, (cal_md.get(card)).split(','))
            # Get the ID or index of the fake object
            fakeID = int(m.group(1))
            fakeList[counts] = [fakeID, x, y]
            counts += 1

    # Match the fake object to the source list
    srcIndex = collections.defaultdict(list)
    for fid, fcoord  in fakeList.items():
        separation = np.sqrt(np.abs(srcX-fcoord[1])**2 +
                             np.abs(srcY-fcoord[2])**2)
        matched = (separation <= tol)
        matchId = np.where(matched)[0]
        matchSp = separation[matchId]
        sortId = [matchId for (matchSp, matchId) in sorted(zip(matchSp,
                                                               matchId))]
        # DEBUG:
        # print fid, fcoord, matchId
        print sortId, sorted(matchSp), matchId
        # Select the index of all matched object
        srcIndex[fid] = sortId

    # Return the source list
    mapper = SchemaMapper(sources.schema)
    mapper.addMinimalSchema(sources.schema)
    newSchema = mapper.getOutputSchema()
    newSchema.addField('fakeId', type=int,
                       doc='id of fake source matched to position')
    srcList = SourceCatalog(newSchema)
    srcList.reserve(sum([len(s) for s in srcIndex.values()]))

    # Return a list of interesting parameters
    #srcParam = collections.defaultdict(list)
    srcParam = []
    nFake = 0
    for matchIndex in srcIndex.values():
        # Check if there is a match
        if len(matchIndex) > 0:
            # Only select the one with the smallest separation
            # TODO: actually get the one with minimum separation
            ss = matchIndex[0]
            fakeObj = fakeList[nFake]
            diffX = srcX[ss] - fakeObj[1]
            diffY = srcY[ss] - fakeObj[2]
            paramList = (fakeObj[0], fakeObj[1], fakeObj[2],
                         magKron[ss], merrKron[ss], magCmod[ss], merrCmod[ss],
                         magExp[ss], merrExp[ss], magDev[ss], merrDev[ss],
                         sdssR[ss], sdssBa[ss], sdssPa[ss],
                         expR[ss], expBa[ss], expPa[ss],
                         devR[ss], devBa[ss], devPa[ss],
                         diffX, diffY, fracDev[ss],
                         parentID[ss], extendClass[ss])
            srcParam.append(paramList)
        else:
            paramList = (fakeObj[0], fakeObj[1], fakeObj[2],
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1,
                         0, -1, -1)
            srcParam.append(paramList)
        # Go to another fake object
        nFake += 1

    # Make a numpy record array
    srcParam = np.array(srcParam, dtype=[('fakeID', int),
                                         ('fakeX', float),
                                         ('fakeY', float),
                                         ('magKron', float),
                                         ('errKron', float),
                                         ('magCmod', float),
                                         ('errCmod', float),
                                         ('magExp', float),
                                         ('errExp', float),
                                         ('magDev', float),
                                         ('errDev', float),
                                         ('sdssR', float),
                                         ('sdssBa', float),
                                         ('sdssPa', float),
                                         ('expR', float),
                                         ('expBa', float),
                                         ('expPa', float),
                                         ('devR', float),
                                         ('devBa', float),
                                         ('devPa', float),
                                         ('diffX', float),
                                         ('diffY', float),
                                         ('fracDev', float),
                                         ('parentID', int),
                                         ('extendClass', float)])

    return srcIndex, srcParam, srcList, zeropoint


def main():

    #TODO: this should use the LSST/HSC conventions
    parser = argparse.ArgumentParser()
    parser.add_argument('rootDir', help='root dir of data repo')
    parser.add_argument('visit',   help='id of visit', type=int)
    parser.add_argument('ccd',     help='id of ccd',   type=int)
    parser.add_argument('tol',     help='tolerence in matching', type=float)
    args = parser.parse_args()

    # Get the information of the fake objects from the output source catalog
    (fakeIndex, fakeParam, fakeList, zp) = getGalaxy(args.rootDir,
                                                               args.visit,
                                                               args.ccd,
                                                               args.tol)

    fakeID   = fakeParam['fakeID']
    magCmod  = fakeParam['magCmod']
    merrCmod = fakeParam['magCmod']
    magKron  = fakeParam['magKron']
    merrKron = fakeParam['magKron']
    magExp  = fakeParam['magExp']
    merrExp = fakeParam['magExp']
    magDev  = fakeParam['magDev']
    merrDev = fakeParam['magDev']
    fracDev  = fakeParam['fracDev']
    parent = fakeParam['parentID']
    fakeX  = fakeParam['fakeX']
    fakeY  = fakeParam['fakeY']
    diffX  = fakeParam['diffX']
    diffY  = fakeParam['diffY']

    # Number of injected fake objects, and the number of the objects recovered
    # by the pipeline (using the selected tol during matching)
    nInject = len(fakeID)
    nMatch  = len(np.argwhere(magCmod))

    # Print out some information
    print '###################################################################'
    print "# Number of Injected Objects : %d" % nInject
    print "# Number of Matched  Objects : %d" % nMatch
    print "# The zeropoint of this CCD is %6.3f" % zp
    print "# Visit = %d   CCD = %d" % (args.visit, args.ccd)
    print '###################################################################'
    print "# FakeX  FakeY  DiffX  DiffY  CModel  CmodelErr  Kron  FracDev  Match  Deblend "

    matchedArr = []
    deblendArr = []
    for i in range(nInject):
       if len(fakeIndex[i]) > 1:
           matched = "multiple"
       elif magCmod[i] > 0:
           matched = "  single"
       else:
           matched = " nomatch"

       if (parent[i] > 0):
           deblend = "deblend"
       else:
           deblend = "isolate"

       matchedArr.append(matched)
       deblendArr.append(deblend)

       print "%6.1d  %6.1d  %6.1f  %6.1f  %7.3f  %6.3f  %7.3f  %4.1f  %s  %s" % (
             fakeX[i], fakeY[i], diffX[i], diffY[i], magCmod[i], merrCmod[i],
             magKron[i], fracDev[i], matched, deblend)

    # Save the results to a fits file
    rootDir = args.rootDir
    if rootDir[-1] is '/':
        rerunName = rootDir.split('/')[-2]
    else:
        rerunName = rootDir.split('/')[-1]
    outFits = rerunName + "_" + str(args.visit) + "_" + str(args.ccd) + "_fakeMatch.fits"
    tabHdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='fakeID', format='I', array=fakeID),
         fits.Column(name='fakeX', format='E', array=fakeX),
         fits.Column(name='fakeY', format='E', array=fakeY),
         fits.Column(name='diffX', format='E', array=diffX),
         fits.Column(name='diffY', format='E', array=diffY),
         fits.Column(name='magCmod',  format='E', array=magCmod),
         fits.Column(name='merrCmod', format='E', array=merrCmod),
         fits.Column(name='magKron',  format='E', array=magKron),
         fits.Column(name='merrKron', format='E', array=merrKron),
         fits.Column(name='magExp',  format='E', array=magExp),
         fits.Column(name='merrExp', format='E', array=merrExp),
         fits.Column(name='magDev',  format='E', array=magDev),
         fits.Column(name='merrDev', format='E', array=merrDev),
         fits.Column(name='sdssR',  format='E', array=fakeParam['sdssR']),
         fits.Column(name='sdssBa', format='E', array=fakeParam['sdssBa']),
         fits.Column(name='sdssPa', format='E', array=fakeParam['sdssPa']),
         fits.Column(name='expR',  format='E', array=fakeParam['expR']),
         fits.Column(name='expBa', format='E', array=fakeParam['expBa']),
         fits.Column(name='expPa', format='E', array=fakeParam['expPa']),
         fits.Column(name='devR',  format='E', array=fakeParam['devR']),
         fits.Column(name='devBa', format='E', array=fakeParam['devBa']),
         fits.Column(name='devPa', format='E', array=fakeParam['devPa']),
         fits.Column(name='fracDev', format='E', array=fracDev),
         fits.Column(name='matched', format='A', array=matchedArr),
         fits.Column(name='deblend', format='A', array=deblendArr)])
    priHdr = fits.Header()
    priHdu = fits.PrimaryHDU(header=priHdr)
    thuList = fits.HDUList([priHdu, tabHdu])
    thuList.writeto(outFits)

if __name__=='__main__':
    main()
