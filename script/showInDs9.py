#!/usr/bin/env python

import argparse
import lsst.daf.persistence  as dafPersist
import lsst.afw.image        as afwImage
import lsst.afw.display.ds9  as ds9

def main(rootDir, visit, ccd, filter=None,
         frame=1, title="", scale="zscale", zoom="to fit", trans=60, useEllipse=False):

    if ccd.find(",") < 0:
        ccd = int(ccd)
        
    # make a butler and specify your dataId
    butler = dafPersist.Butler(rootDir)
    if filter:
        dataId = {'tract': visit, 'patch':ccd, 'filter': filter}
        exposure = butler.get("deepCoadd", dataId)
        butlerTarget='deepCoadd_'
    else:
        dataId = {'visit': visit, 'ccd':ccd}
        exposure = butler.get("calexp", dataId)
        butlerTarget=""
    

    # put the settings in a dict object and call ds9.mtv()
    settings = {'scale':scale, 'zoom': zoom, 'mask' : 'transparency %d' %(trans)}
    ds9.mtv(exposure, frame=frame, title=title, settings=settings)


    # now get the source catalog and overplot the points

    sources = butler.get(butlerTarget+'src', dataId)

    with ds9.Buffering():
        print len(sources)
        x0, y0 = exposure.getXY0()
        print x0, y0
        for i,source in enumerate(sources):
            color = ds9.RED
            size = 5.0
            
            if useEllipse:
                # show an ellipse symbol
                symbol = "@:{ixx},{ixy},{iyy}".format(ixx=source.getIxx(),
                                                      ixy=source.getIxy(),
                                                      iyy=source.getIyy())
            else:
                # just a simple point (symbols +, x, *, o are all accepted)
                symbol = "o"
                
            ds9.dot(symbol, source.getX()-x0, source.getY()-y0, ctype=color,
                    size=size, frame=frame, silent=True)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("visit", type=int, help="Visit to show")
    parser.add_argument("ccd", help="CCD to show")
    parser.add_argument("-F", "--filter", default=None, help="Use visit,ccd as tract,patch ... add filter")
    parser.add_argument("-e", "--useEllipse", default=False, action='store_true', help="Overplot ellipses")
    parser.add_argument("-f", "--frame", type=int, default=1, help="Frame")
    parser.add_argument("-s", "--scale", default="zscale", help="Gray-scale")
    parser.add_argument("-t", "--title", default="", help="Figure title")
    parser.add_argument("-T", "--trans", default=60, help="Transparency")
    parser.add_argument("-z", "--zoom",  default="to fit", help="Zoom")
    args = parser.parse_args()

    main(args.root, args.visit, args.ccd, filter=args.filter,
         frame=args.frame, title=args.title, scale=args.scale, zoom=args.zoom, trans=args.trans,
         useEllipse=args.useEllipse
     )
