import numpy as np
import statsmodels.api as sm

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

        outlier : numpy array
                 Default value False. Set to True if the observation 
                  is outlier
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
                self.outlier = np.repeat(False, len(self.t))

        def lowessClean(self, K=5):
                lowess = sm.nonparametric.lowess
                z = lowess(self.m, self.t,frac=0.4)
                residuals = z[:,1]-self.m
                residSigma = np.std(residuals)
                self.outlier = np.abs(residuals) > K*residSigma
                
