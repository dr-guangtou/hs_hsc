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
Tests for MultiGaussianObjective

Run with:
   ./testMultiGaussianObjective.py
or
   python
   >>> import testMultiGaussianObjective; testMultiGaussianObjective.run()
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
numpy.set_printoptions(linewidth=110)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MultiGaussianObjectiveTestCase(unittest.TestCase):

    def assertClose(self, a, b, rtol=1E-5, atol=1E-8):
        self.assert_(numpy.allclose(a, b, rtol=rtol, atol=atol), "\n%s\n!=\n%s" % (a, b))

    def buildNumericalDerivative(self, builder, parameters, makeEllipse):
        eps = 1E-6
        derivative = numpy.zeros((len(parameters), builder.getSize()), dtype=float).transpose()
        for i in range(len(parameters)):
            parameters[i] += eps
            ellipse = makeEllipse(parameters)
            builder.update(ellipse)
            derivative[:,i] = builder.getModel()
            parameters[i] -= 2.0 * eps
            ellipse = makeEllipse(parameters)
            builder.update(ellipse)
            derivative[:,i] -= builder.getModel()
            derivative[:,i] /= 2.0 * eps
        return derivative

    def setUp(self):
        self.ellipse = ellipses.Axes(10, 7, 0.3)
        self.center = geom.Point2D(10.1, 11.2)
        self.bbox = geom.Box2I(geom.Point2I(5, 6), geom.Point2I(13, 12))
        self.image = lsst.afw.image.ImageF(self.bbox)
        x, y = numpy.meshgrid(
            numpy.arange(self.bbox.getBeginX(), self.bbox.getEndX()),
            numpy.arange(self.bbox.getBeginY(), self.bbox.getEndY())
            )
        # use exponential profile to ensure we don't get exact fit
        self.image.getArray()[:,:] = numpy.exp(-(x**2 + y**2)**0.5)
        self.inputs = ms.ModelInputHandler(self.image, self.center, self.bbox)

    def tearDown(self):
        del self.ellipse
        del self.bbox
        del self.image
        del self.center
        del self.inputs

    def doTest(self, multiGaussian):
        eps = 1E-6
        obj = ms.MultiGaussianObjective(self.inputs, multiGaussian)
        parameters = numpy.array(
            [[0.1, 0.2, 3.0],
             [0.0, 0.0, 4.0],
             [0.8, -1.1, 5.0]]
            )
        for i in range(parameters.shape[0]):
            d0 = numpy.zeros((parameters.shape[1], self.inputs.getSize()), dtype=float).transpose()
            d1 = numpy.zeros((parameters.shape[1], self.inputs.getSize()), dtype=float).transpose()
            f0 = numpy.zeros(self.inputs.getSize(), dtype=float)
            obj.computeFunction(parameters[i,:], f0)
            x, residuals, rank, s = numpy.linalg.lstsq(
                obj.getModel().reshape(f0.size, 1),
                self.inputs.getData()
                )
            self.assertClose(x, [obj.getAmplitude()])
            self.assertClose(obj.getAmplitude() * obj.getModel() - self.inputs.getData(), f0)
            obj.computeDerivative(parameters[i,:], f0, d0)
            for j in range(parameters.shape[1]):
                parameters[i,j] += eps
                f1a = numpy.zeros(self.inputs.getSize(), dtype=float)
                obj.computeFunction(parameters[i,:], f1a)
                parameters[i,j] -= 2.0*eps
                f1b = numpy.zeros(self.inputs.getSize(), dtype=float)
                obj.computeFunction(parameters[i,:], f1b)
                d1[:,j] = (f1a - f1b) / (2.0 * eps)
                parameters[i,j] += eps
            self.assertClose(d0, d1, rtol=1E-10, atol=1E-8)

    def testUnconvolved(self):
        multiGaussian = ms.MultiGaussian()
        multiGaussian.add(ms.GaussianComponent(1.0, 1.0))
        multiGaussian.add(ms.GaussianComponent(1.23, 1.32))
        multiGaussian.add(ms.GaussianComponent(0.67, 0.9))
        self.doTest(multiGaussian)
        

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MultiGaussianObjectiveTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
