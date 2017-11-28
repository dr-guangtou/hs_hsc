#!/usr/bin/env python
# encoding: utf-8
"""Generate cutout images in batch mode."""

import os
import argparse
import coaddBatchCutout as cbc

COM = '#' * 100
SEP = '-' * 100
WAR = '!' * 100


def run(args):
    """Run cutout in batch mode."""
    if args.noZCutout:
        zCut = False
    else:
        zCut = True

    if os.path.isfile(args.incat):
        cbc.coaddBatchCutFull(args.root, args.incat,
                              size=args.size,
                              band=args.band,
                              idField=args.idField,
                              prefix=args.prefix,
                              zCutoutSize=zCut,
                              zField=args.zField,
                              onlyColor=args.onlyColor,
                              noColor=args.noColor,
                              colorFilters=args.colorFilters,
                              infoField1=args.infoField1,
                              infoField2=args.infoField2,
                              saveSrc=False,
                              makeDir=args.makeDir,
                              raField=args.raField,
                              decField=args.decField,
                              clean=args.clean,
                              njobs=args.njobs,
                              sample=args.sample)
    else:
        raise Exception("### Can not find the input catalog: %s" % args.incat)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("incat", help="The input catalog for cutout")
    parser.add_argument("-s", '--size', dest='size', type=int,
                        help="Half size of the cutout box", default=200)
    parser.add_argument('-f', '--filter', dest='band', help="Filter",
                        default='HSC-I')
    parser.add_argument('-cf', '--color-filters', dest='colorFilters',
                        help="Choice of filters for color images",
                        default='gri')
    parser.add_argument('-sf', '--size-field', dest='sizeField',
                        help="Column name for cutout size",
                        default='cutout_size')
    parser.add_argument('-info1', '--infoField1', dest='infoField1',
                        help="Column name for first extra information",
                        default='MSTAR')
    parser.add_argument('-info2', '--infoField2', dest='infoField2',
                        help="Column name for second extra information",
                        default='VDISP')
    parser.add_argument('-oc', '--onlyColor', action="store_true",
                        dest='onlyColor', default=False)
    parser.add_argument('-safe', '--safe', action="store_true", dest='safe',
                        default=False)
    parser.add_argument('-clean', '--clean', action="store_true", dest='clean',
                        default=False)
    parser.add_argument('-v', '--verbose', action="store_true", dest='verbose',
                        default=False)
    parser.add_argument('-ns', '--noSrc', action="store_false", dest='saveSrc',
                        default=True)
    parser.add_argument('-makeDir', '--makeDir', action="store_true",
                        dest='makeDir', default=True)
    parser.add_argument('-nz', '--noZCutout', action="store_true",
                        dest='noZCutout', default=False)
    parser.add_argument('-nc', '--noColor', action="store_true",
                        dest='noColor', default=True)
    parser.add_argument('-p', '--prefix', dest='prefix',
                        help='Prefix of the output file',
                        default='hsc')
    parser.add_argument('-id', '--id', dest='idField',
                        help="Column name for ID", default='index')
    parser.add_argument('-ra', '--ra', dest='raField',
                        help="Column name for RA", default='ra_hsc')
    parser.add_argument('-dec', '--dec', dest='decField',
                        help="Column name for DEC", default='dec_hsc')
    parser.add_argument('-z', '--redshift', dest='zField',
                        help="Column name for z", default='z_use')
    parser.add_argument('-j', '--njobs', type=int,
                        help='Number of jobs run at the same time',
                        dest='njobs', default=1)
    parser.add_argument('--sample', dest='sample', help="Sample name",
                        default=None)

    args = parser.parse_args()

    run(args)
