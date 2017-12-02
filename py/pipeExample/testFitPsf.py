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

"""
Tests for FitPsf

Run with:
   ./testFitPsf.py
or
   python
   >>> import testFitPsf; testFitPsf.run()
"""

import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.geom as geom
import lsst.afw.geom.ellipses as ellipses
import lsst.afw.image
import lsst.afw.detection
import lsst.meas.extensions.multiShapelet as ms

numpy.random.seed(5)
numpy.set_printoptions(linewidth=120)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class FitPsfTestCase(unittest.TestCase):

    def assertClose(self, a, b, rtol=1E-5, atol=1E-8):
        self.assert_(numpy.allclose(a, b, rtol=rtol, atol=atol), "\n%s\n!=\n%s" % (a, b))

    def testObjective(self):
        eps = 1E-6
        ctrl = ms.FitPsfControl()
        image = lsst.afw.image.ImageD(5, 5)
        nParameters = 3
        nData = image.getBBox().getArea()
        nTests = 10
        center = geom.Point2D(2.0, 2.0)
        xGrid, yGrid = numpy.meshgrid(numpy.arange(-2, 3), numpy.arange(-2, 3))
        image.getArray()[:,:] = 1.0 * numpy.exp(-0.5*(xGrid**2 + yGrid**2))
        image.getArray()[:,:] += numpy.random.randn(5, 5) * 0.1
        inputs = ms.ModelInputHandler(image, center, image.getBBox(lsst.afw.image.PARENT))
        obj = ms.FitPsfAlgorithm.makeObjective(ctrl, inputs)
        parameters = numpy.random.rand(nTests, nParameters) * 0.5
        for i in range(nTests):
            f0 = numpy.zeros(nData, dtype=float)
            obj.computeFunction(parameters[i,:], f0)
            f1 = obj.getModel() * obj.getAmplitude() - inputs.getData()
            model = ms.FitPsfModel(ctrl, obj.getAmplitude(), parameters[i,:])
            self.assertClose(model.outer[0], ctrl.peakRatio * model.inner[0] * ctrl.radiusRatio**2)
            self.assertEqual(model.radiusRatio, ctrl.radiusRatio)
            image2 = lsst.afw.image.ImageD(5, 5)
            multiShapeletFunc = model.asMultiShapelet(center)
            multiShapeletFunc.evaluate().addToImage(image2)
            f2 = (image2.getArray().ravel() - inputs.getData())
            multiGaussian = model.getMultiGaussian()
            builder1 = ms.GaussianModelBuilder(inputs.getX(), inputs.getY(),
                                               multiGaussian[0].flux, multiGaussian[0].radius)
            builder2 = ms.GaussianModelBuilder(inputs.getX(), inputs.getY(),
                                               multiGaussian[1].flux, multiGaussian[1].radius)
            builder1.update(model.ellipse)
            builder2.update(model.ellipse)
            f3 = builder1.getModel() + builder2.getModel() - inputs.getData()
            self.assertClose(f0, f1)
            self.assertClose(f0, f2)
            self.assertClose(f0, f3)
            d0 = numpy.zeros((nParameters, nData), dtype=float).transpose()
            d1 = numpy.zeros((nParameters, nData), dtype=float).transpose()
            obj.computeDerivative(parameters[i,:], f0, d0)
            for j in range(nParameters):
                parameters[i,j] += eps
                f1a = numpy.zeros(nData, dtype=float)
                obj.computeFunction(parameters[i,:], f1a)
                parameters[i,j] -= 2.0*eps
                f1b = numpy.zeros(nData, dtype=float)
                obj.computeFunction(parameters[i,:], f1b)
                d1[:,j] = (f1a - f1b) / (2.0 * eps)
                parameters[i,j] += eps
            self.assertClose(d0, d1, rtol=1E-10, atol=1E-8)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(FitPsfTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
