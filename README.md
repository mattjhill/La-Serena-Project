# La-Serena-Project
La Serena Data Science School Project on finding periodic signals in CRTS quasar light curves

# Dependencies

The scripts require `gatspy` (for fast Lomb-Scargle) and `pyaov` (for Analysis of Variance).

# Usage

The code is built around the `LightCurve` object, which is instantiated from a three column data file (time, magnitude, magnitude error).

	>>> from lightcurve import LightCurve
	>>> lc = LightCurve("/my/lightcurve.dat") #the file containing the lightcurve
	>>> lc.get_pbest()
	Peak of    1.350E+02 at frequency       3.64061161304
	0.610526068299 
