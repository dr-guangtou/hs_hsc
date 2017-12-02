# Image Analysis Tools for Hyper Suprime-Cam Survey

----

* Ok, here are some Python codes I wrote (from three years ago) to deal with the internal data release of Hyper Suprime-Cam Subaru Strategic Program while I was a post-doc at IPMU.

* After selling my soul to IDL for my entire PhD, I started to learn Python by working on HSC data.  So, yes, I am embarrassed by these codes, and I am slowly turning into more Python-like style.  

* These codes are used in the following publications:
    - [Individual Stellar Halos of Massive Galaxies Measured to 100 kpc at 0.3<z<0.5 using Hyper Suprime-Cam](http://adsabs.harvard.edu/abs/2017arXiv170701904H)
    - A Detection of the Environmental Dependence of the Sizes and Stellar Haloes
       of Massive Central Galaxies (Submitted)

* If you have access to the internal HSC data, these codes can help you generate cutout images, make three-color pictures, figure out which part of the data is useful.

* They can also help you perform old-fashion 1-D photometry or fit galaxy using GALFIT.

* It is highly recommended that you stay out of this mess, but if somehow you are convinced that they can be used to you, please feel free to let me know.

* If you want to use HSC data, and you are patient enough, I am also learning how to write (somewhat) decent Python packages to:
    1. Search and download public- or internal HSC data: [unagi](https://github.com/dr-guangtou/unagi)
    2. Photometry for single galaxy: [kungpao](https://github.com/dr-guangtou/kungpao)
    - Comments or advice or help are highly welcomed!
    - And, yes, I name all my codes with my favorite food, and they are acronym of nothing.
