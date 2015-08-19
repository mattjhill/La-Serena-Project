import numpy as np

class LightCurve(object):
	""" 
	The CRTS light curve object 

	Attributes
	----------
	t : numpy array
		the Modified Julian Dates, i.e. the time of observation
	m : numpy array
		the mesured magnitude at the time t
	merr : numpy array
		the photometric magnitude error for the observation
	"""

	def __init__(self, fname):
		""" 
		The initilization function. It creates an object from
		the actual CRTS data file.

		Parameters
		----------
		fname : string
			the filename of the light curve
		"""
		self.t, self.m, self.merr = np.loadtxt(fname, unpack=True)
	