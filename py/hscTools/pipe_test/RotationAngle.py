#!/usr/bin/env python

# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
# 
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import lsst.pex.config as pexConf
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDetection
import lsst.meas.algorithms as measAlg
import lsst.utils.tests as utilsTests

import lsst.meas.extensions.rotAngle
import numpy

try:
    type(verbose)
except NameError:
    verbose = 0

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class RotationAngleTestCase(unittest.TestCase):
    """A test case for rotation angle"""

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def measureRotAngle(self, exposure, x, y):
        """Measure rotation angle quantities using the C++ code"""
        msConfig = measAlg.SourceMeasurementConfig()
        msConfig.algorithms.names.add("rotAngle")
        schema = afwTable.SourceTable.makeMinimalSchema()
        ms = msConfig.makeMeasureSources(schema)
        table = afwTable.SourceTable.make(schema)
        msConfig.slots.setupTable(table)
        source = table.makeRecord()
        fp = afwDetection.Footprint(exposure.getBBox())
        source.setFootprint(fp)
        center = afwGeom.Point2D(x, y)
        ms.apply(source, exposure, center)

        return source.get("rotAngle.north"), source.get("rotAngle.east")


    def testMeasure(self):
        """Test that we can instantiate and play with RotationAngle"""

        scale = 5.0e-5                  # Pixel scale: 0.18 arcsec/pixel
        for angle in (0, 45, 90, 120, 180, 275):
            angle = angle * afwGeom.degrees
            for expFactory in (afwImage.ExposureF, afwImage.ExposureD):
                cdMatrix = numpy.array([[scale * math.cos(angle.asRadians()), 
                                         scale * math.sin(angle.asRadians())],
                                        [scale * -math.sin(angle.asRadians()),
                                         scale * math.cos(angle.asRadians())]])
                
                wcs = afwImage.Wcs(afwGeom.Point2D(0, 0), afwGeom.Point2D(50, 50), cdMatrix)
                exp = expFactory(afwGeom.ExtentI(100, 100), wcs)
                x, y = 10, 20
                exp.getMaskedImage().getImage().set(x, y, 1.0)

                east, north = self.measureRotAngle(exp, x, y)

                eastTrue = angle
                northTrue = angle + 90 * afwGeom.degrees
                while east < 0 * afwGeom.degrees: east = east + 360 * afwGeom.degrees
                while north < 0 * afwGeom.degrees: north = north + 360 * afwGeom.degrees
                while northTrue > 360 * afwGeom.degrees: northTrue = northTrue - 360 * afwGeom.degrees

                self.assertAlmostEqual(east.asRadians(), eastTrue.asRadians())
                self.assertAlmostEqual(north.asRadians(), northTrue.asRadians())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(RotationAngleTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)
 
if __name__ == "__main__":
    run(True)
