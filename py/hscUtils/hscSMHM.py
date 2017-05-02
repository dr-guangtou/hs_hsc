"""
Parameterized models of the stellar mass - halo mass relation (SMHM).

"""

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import numpy as np

__all__ = ['Leauthaud12']

__author__ = "Song Huang"
__copyright__ = "Copyright 2017, All mankind"
__email__ = "shuang89@ucsc.edu"


class Behroozi10():
    """
    Class for stellar mass halo mass relation from Behroozi+10
    """
    def __init__(self, redshift=0.3):
        """Initiallize the model."""

    def toMhalo(self, Ms):
        """Estimate halo mass via stellar mass."""
        return None


class Leauthaud12():
    """
    Class for stellar mass halo mass relation from Leauthaud+2012.
    """

    def __init__(self, redshift=0.3, sigmod=1):
        """Initiallize the model."""
        if (redshift > 0.22) and (redshift <= 0.48):

            if sigmod == 1:

                self.param = {'logM1': 12.520, 'logM1Err': 0.037,
                              'logMs0': 10.916, 'logMs0Err': 0.020,
                              'beta': 0.457, 'betaErr': 0.009,
                              'delta': 0.566, 'deltaErr': 0.086,
                              'gamma': 1.530, 'gammaErr': 0.180,
                              'sigMs': 0.206, 'sigMsErr': 0.031,
                              'bCut': 1.47, 'bCutErr': 0.73,
                              'bSat': 10.62, 'bSatErr': 0.87,
                              'betaCut': -0.13, 'betaCutErr': 0.28,
                              'betaSat': 0.859, 'betaSatErr': 0.038}

            elif sigmod == 2:

                self.param = {'logM1': 12.518, 'logM1Err': 0.038,
                              'logMs0': 10.917, 'logMs0Err': 0.020,
                              'beta': 0.456, 'betaErr': 0.009,
                              'delta': 0.582, 'deltaErr': 0.083,
                              'gamma': 1.480, 'gammaErr': 0.170,
                              'sigMs': 0.192, 'sigMsErr': 0.031,
                              'bCut': 1.52, 'bCutErr': 0.79,
                              'bSat': 10.69, 'bSatErr': 0.89,
                              'betaCut': -0.11, 'betaCutErr': 0.29,
                              'betaSat': 0.860, 'betaSatErr': 0.039}

            else:
                raise KeyError("Wrong SIG_MOD choice!!")

        elif (redshift > 0.48) and (redshift <= 0.74):

            if sigmod == 1:

                self.param = {'logM1': 12.725, 'logM1Err': 0.032,
                              'logMs0': 11.038, 'logMs0Err': 0.019,
                              'beta': 0.466, 'betaErr': 0.009,
                              'delta': 0.610, 'deltaErr': 0.130,
                              'gamma': 1.950, 'gammaErr': 0.250,
                              'sigMs': 0.249, 'sigMsErr': 0.019,
                              'bCut': 1.65, 'bCutErr': 0.65,
                              'bSat': 9.04, 'bSatErr': 0.81,
                              'betaCut': 0.590, 'betaCutErr': 0.280,
                              'betaSat': 0.740, 'betaSatErr': 0.059}

            else:
                # TODO: Parameters for sigmod=2
                raise KeyError("Wrong SIG_MOD choice!!")

        elif (redshift > 0.74) and (redshift <= 1.00):

            if sigmod == 1:

                self.param = {'logM1': 12.722, 'logM1Err': 0.027,
                              'logMs0': 11.100, 'logMs0Err': 0.018,
                              'beta': 0.470, 'betaErr': 0.008,
                              'delta': 0.393, 'deltaErr': 0.088,
                              'gamma': 2.510, 'gammaErr': 0.250,
                              'sigMs': 0.227, 'sigMsErr': 0.020,
                              'bCut': 2.46, 'bCutErr': 0.53,
                              'bSat': 8.72, 'bSatErr': 0.53,
                              'betaCut': 0.570, 'betaCutErr': 0.200,
                              'betaSat': 0.863, 'betaSatErr': 0.053}

            else:
                # TODO: Parameters for sigmod=2
                raise KeyError("Wrong SIG_MOD choice!!")

        else:
            raise KeyError("Wrong Redshift choice!!")

    def toMstar(self, Mh):
        """Estimate stellar mass via halo mass."""
        # TODO: Place holder

        return None

    def toMhalo(self, Ms):
        """Estimate halo mass via stellar mass."""
        param = self.param

        mRatio = Ms / (10.0 ** param['logMs0'])

        termB = np.log10(mRatio) * param['beta']
        termC = mRatio ** param['delta']
        termD = mRatio ** (param['gamma'] * -1.0) + 1.0

        logMh = param['logM1'] + termB + (termC / termD) - 0.50

        return logMh

    def getMhalo(self, Ms, m0=10.92, m1=12.52, beta=0.457, delta=0.566,
                 gamma=1.530):
        """Estimate halo mass via stellar mass using specific parameters."""
        mRatio = Ms / (10.0 ** m0)

        termB = np.log10(mRatio) * beta
        termC = mRatio ** delta
        termD = mRatio ** (gamma * -1.0) + 1.0

        logMh = m1 + termB + (termC / termD) - 0.50

        return logMh

    def bootstrapMhalo(self, Ms, n=1000):
        """Bootsrtap the uncertainties of the SMHM parameters."""
        param = self.param

        m0 = np.random.normal(param['logMs0'], param['logMs0Err'], n)
        m1 = np.random.normal(param['logM1'], param['logM1Err'], n)
        bb = np.random.normal(param['beta'], param['betaErr'], n)
        dd = np.random.normal(param['delta'], param['deltaErr'], n)
        gg = np.random.normal(param['gamma'], param['gammaErr'], n)

        mhArr = np.asarray([self.getMhalo(Ms, m0=m0[ii], m1=m1[ii],
                                          beta=bb[ii], delta=dd[ii],
                                          gamma=gg[ii])
                            for ii in range(n)])

        return mhArr
