import sys
from lightcurve import LightCurve

"""Takes a file name of the a lightcurve as the command line arg.  For running on the hpc"""
fname = '/home/apps/astro/DATA/CRTSQSO'+sys.argv[1][1:]
lc = LightCurve(fname)
lc.analyze("/home/uchile/cmm/astrolab/student01/mjh/outfiles/"+fname.split('/')[-1])