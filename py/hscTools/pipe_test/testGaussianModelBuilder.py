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
Tests for GaussianModelBuilder

Run with:
   ./testGaussianModelBuilder.py
or
   python
   >>> import testGaussianModelBuilder; testGaussianModelBuilder.run()
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

class GaussianModelBuilderTestCase(unittest.TestCase):

    def assertClose(self, a, b, rtol=1E-5, atol=1E-8):
        self.assert_(numpy.allclose(a, b, rtol=rtol, atol=atol), "\n%s\n!=\n%s" % (a, b))

    def buildModel(self, ellipse):
        LT = geom.LinearTransform
        gt = ellipse.getGridTransform()
        xt = gt[LT.XX] * self.xg + gt[LT.XY] * self.yg
        yt = gt[LT.YX] * self.xg + gt[LT.YY] * self.yg
        model = numpy.exp(-0.5 * (yt**2 + xt**2)) / (ellipse.getArea() * 2.0)
        return model.ravel()

    def evalShapelets(self, func):
        z = numpy.zeros(self.x.size, dtype=float)
        ev = func.evaluate()
        for i in range(self.x.size):
            z[i] = ev(self.x[i], self.y[i])
        return z

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
        self.xg, self.yg = numpy.meshgrid(numpy.linspace(-20, 20, 101), numpy.linspace(-15, 25, 95))
        self.x = self.xg.ravel()
        self.y = self.yg.ravel()

    def tearDown(self):
        del self.ellipse

    def testModel1(self):
        builder = ms.GaussianModelBuilder(self.x, self.y)
        builder.update(self.ellipse)
        mgc = ms.GaussianComponent()
        shapelet = mgc.makeShapelet(ellipses.Ellipse(self.ellipse))
        z0 = builder.getModel()
        z1 = self.buildModel(self.ellipse)
        z2 = self.evalShapelets(shapelet)
        self.assertClose(z0, z1)
        self.assertClose(z0, z2)

    def testModel2(self):
        amplitude = 3.2
        radius = 2.7
        mgc = ms.GaussianComponent(amplitude, radius)
        shapelet = mgc.makeShapelet(ellipses.Ellipse(self.ellipse))
        builder = ms.GaussianModelBuilder(self.x, self.y, amplitude, radius)
        builder.update(self.ellipse)
        self.ellipse.scale(radius)
        z0 = builder.getModel()
        z1 = amplitude * self.buildModel(self.ellipse)
        z2 = self.evalShapelets(shapelet)
        self.assertClose(z0, z1)
        self.assertClose(z0, z2)
        
    def testModel3(self):
        amplitude = 3.2
        radius = 2.7
        psfEllipse = ellipses.Quadrupole(2.3, 1.8, 0.6)
        psfAmplitude = 5.3
        mgc = ms.GaussianComponent(amplitude, radius)
        shapelet = mgc.makeShapelet(ellipses.Ellipse(self.ellipse))
        psf = ms.GaussianComponent(psfAmplitude, 1.0).makeShapelet(ellipses.Ellipse(psfEllipse))
        shapelet = shapelet.convolve(psf)
        builder = ms.GaussianModelBuilder(self.x, self.y, amplitude, radius, psfEllipse, psfAmplitude)
        builder.update(self.ellipse)
        self.ellipse.scale(radius)
        ellipse = self.ellipse.convolve(psfEllipse)
        self.assertClose(builder.getModel(), amplitude * psfAmplitude * self.buildModel(ellipse))
        z0 = builder.getModel()
        z1 = psfAmplitude * amplitude * self.buildModel(ellipse)
        z2 = self.evalShapelets(shapelet)
        self.assertClose(z0, z1)
        self.assertClose(z0, z2)

    def testDerivative1(self):
        builder = ms.GaussianModelBuilder(self.x, self.y)
        a = numpy.zeros((3, builder.getSize()), dtype=float).transpose()
        builder.update(self.ellipse)
        builder.computeDerivative(a)
        def makeEllipse(p):
            return ellipses.Axes(*p)
        n = self.buildNumericalDerivative(builder, self.ellipse.getParameterVector(), makeEllipse)
        # no hard requirement for tolerances here, but I've dialed them to the max to avoid regressions
        self.assertClose(a, n, rtol=1E-15, atol=1E-9)

    def testDerivative2(self):
        amplitude = 3.2
        radius = 2.7
        builder = ms.GaussianModelBuilder(self.x, self.y, amplitude, radius)
        a = numpy.zeros((3, builder.getSize()), dtype=float).transpose()
        builder.update(self.ellipse)
        builder.computeDerivative(a)
        def makeEllipse(p):
            return ellipses.Axes(*p)
        n = self.buildNumericalDerivative(builder, self.ellipse.getParameterVector(), makeEllipse)
        # no hard requirement for tolerances here, but I've dialed them to the max to avoid regressions
        self.assertClose(a, n, rtol=1E-15, atol=1E-9)

    def testDerivative3(self):
        amplitude = 3.2
        radius = 2.7
        psfEllipse = ellipses.Quadrupole(2.3, 1.8, 0.6)
        psfAmplitude = 5.3
        builder = ms.GaussianModelBuilder(self.x, self.y, amplitude, radius, psfEllipse, psfAmplitude)
        a = numpy.zeros((3, builder.getSize()), dtype=float).transpose()
        builder.update(self.ellipse)
        builder.computeDerivative(a)
        def makeEllipse(p):
            return ellipses.Axes(*p)
        n = self.buildNumericalDerivative(builder, self.ellipse.getParameterVector(), makeEllipse)
        # no hard requirement for tolerances here, but I've dialed them to the max to avoid regressions
        self.assertClose(a, n, rtol=1E-15, atol=1E-9)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(GaussianModelBuilderTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
