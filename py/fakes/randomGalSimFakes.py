import numpy as np

import lsst.afw.image
import lsst.afw.geom
import lsst.afw.math
import lsst.afw.cameraGeom
import lsst.pex.config
from lsst.pipe.tasks.fakes import FakeSourcesConfig, FakeSourcesTask
import pyfits as fits

import makeFakeGalaxy as makeFake


class RandomGalSimFakesConfig(FakeSourcesConfig):
    galList = lsst.pex.config.Field(dtype=str, doc="catalog of galaxies to add")
    margin = lsst.pex.config.Field(dtype=int, default=None, optional=True,
                                   doc="Size of margin at edge that should not be added")
    seed = lsst.pex.config.Field(dtype=int, default=1,
                                 doc="Seed for random number generator")
    galType = lsst.pex.config.ChoiceField(dtype=str, default='sersic',
                                          allowed={'dsersic':'double sersic galaxies added',
                                                   'sersic':'single sersic galaxies added',
                                                   'real':'real HST galaxy images added'},
                                          doc='type of GalSim galaxies to add')


class RandomGalSimFakesTask(FakeSourcesTask):
    ConfigClass = RandomGalSimFakesConfig

    def __init__(self, **kwargs):
        FakeSourcesTask.__init__(self, **kwargs)
        print "RNG seed:", self.config.seed
        self.rng = lsst.afw.math.Random(self.config.seed)
        self.npRand = np.random.RandomState(self.config.seed)
        self.galData = fits.open(self.config.galList)[1].data


    def run(self, exposure, sources, background):

        self.log.info("Adding fake random galaxies")
        psf = exposure.getPsf()
        psfBBox = psf.computeImage().getBBox()
        minMargin =  max(psfBBox.getWidth(), psfBBox.getHeight())/2 + 1
        md = exposure.getMetadata()
        expBBox = exposure.getBBox()

        for igal, gal in enumerate(self.galData):

            try:
                galident = gal["ID"]
            except KeyError:
                galident = igal + 1

            try:
                flux = exposure.getCalib().getFlux(float(gal['mag']))
            except KeyError:
                raise KeyError("No mag column in %s table"%self.config.galList)

            #don't put the galaxy within one PSF box of the edge
            #or within the given pixel margin
            if self.config.margin is not None:
                margin = self.config.margin
            else:
                margin = minMargin
            bboxI = (exposure.getBBox(lsst.afw.image.PARENT))
            bboxI.grow(-margin)
            bboxD = lsst.afw.geom.BoxD(bboxI)
            x = self.rng.flat(bboxD.getMinX(), bboxD.getMaxX())
            y = self.rng.flat(bboxD.getMinY(), bboxD.getMaxY())
            #TODO: check for overlaps here and regenerate x,y if necessary

            psfImage = psf.computeKernelImage(lsst.afw.geom.Point2D(x, y))

            galArray = makeFake.makeGalaxy(flux, gal, psfImage.getArray(),
                                           self.config.galType )
            galImage = lsst.afw.image.ImageF(galArray.astype(np.float32))
            galBBox = galImage.getBBox(lsst.afw.image.PARENT)
            galImage = lsst.afw.math.offsetImage(galImage,
                                                 x - galBBox.getWidth()/2.0 + 0.5,
                                                 y - galBBox.getHeight()/2.0 + 0.5,
                                                 'lanczos3')
            galBBox = galImage.getBBox(lsst.afw.image.PARENT)


           #check that we're within the larger exposure, otherwise crop
            if expBBox.contains(galImage.getBBox(lsst.afw.image.PARENT)) is False:
                newBBox = galImage.getBBox(lsst.afw.image.PARENT)
                newBBox.clip(expBBox)
                self.log.info("Cropping FAKE%d from %s to %s"%(galident, str(galBBox), str(newBBox)))
                galImage = galImage.Factory(galImage, newBBox, lsst.afw.image.PARENT)
                galBBox = newBBox

            detector = exposure.getDetector()
            ccd =  lsst.afw.cameraGeom.cast_Ccd(detector)
            amp = ccd.findAmp(lsst.afw.geom.Point2I(int(x), int(y)))
            gain = amp.getElectronicParams().getGain()
            #TODO: this is gaussian noise right now, probably good enough
            varImage = lsst.afw.image.ImageF(galImage, True)
            varImage /= gain
            noiseArray = self.npRand.normal(loc=0.0,
                                            scale=np.sqrt(np.abs(varImage.getArray())),
                                             size=(galBBox.getHeight(),
                                                   galBBox.getWidth()))
            noiseImage = lsst.afw.image.ImageF(noiseArray.astype(np.float32))
            galImage += noiseImage

            md.set("FAKE%d" % gal['ID'], "%.3f, %.3f" % (x, y))
            self.log.info("Adding fake at: %.1f,%.1f"% (x, y))

            galMaskedImage = lsst.afw.image.MaskedImageF(galImage, None, varImage)
            #TODO: set the mask
            galMaskedImage.getMask().set(self.bitmask)
            subMaskedImage = exposure.getMaskedImage().Factory(exposure.getMaskedImage(),
                                                               galMaskedImage.getBBox(lsst.afw.image.PARENT),
                                                               lsst.afw.image.PARENT)
            subMaskedImage += galMaskedImage

