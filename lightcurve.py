import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from gatspy.periodic import LombScargleFast
import pyaov


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

    def psearch(self):
        """
        Convole the two periodograms and find the Max
        """
        model = LombScargleFast().fit(self.t, self.m, self.merr)
        self.periods, self.power = model.periodogram_auto(nyquist_factor=200)
        self.aov, self.fr, _ = pyaov.aovw(self.t, self.m, self.merr, fstop=max(1/self.periods), fstep=1/self.periods[0])
        self.aov = self.aov[1:]
        self.fr = self.fr[1:]
        print len(self.aov), len(self.power)

        self.conv_pgram = self.power*self.aov
        self.conv_pgram_max = max(self.conv_pgram) 
        self.conv_fbest = self.fr[np.argmax(self.conv_pgram)]
        self.conv_pbest = 1/self.conv_fbest

        self.pbest_signif = (self.conv_pgram_max - np.median(self.conv_pgram))/np.std(self.conv_pgram)
        print("best period at {:.3f} days, {:.2f} sigma from the median".format(self.conv_pbest, self.pbest_signif))

    def obs_unique(self):
        """
        Average the observation by unique MJD
        Both m and merr are averaged
        """
        t_tmp = np.floor(self.t)
        m_new = np.repeat(0.,len(np.unique(t_tmp)))
        merr_new = m_new.copy()
        j = 0
        for x in sorted(np.unique(t_tmp)):
            m_new[j] = np.average(self.m[np.where(t_tmp == x)])
            merr_new[j] = np.sqrt(np.average(self.merr[np.where(t_tmp==x)]**2))
            j += 1
        self.m = m_new
        self.merr = merr_new

    def lowessClean(self, threshold=0.25):
        lowess = sm.nonparametric.lowess
        z = lowess(self.m, self.t,frac=0.3)
        residuals = z[:,1]-self.m
        residSigma = np.std(residuals)
        self.outlier = np.abs(residuals) > threshold

    def analyze(self):
        self.lowessClean()
        self.psearch()


    # plot observation
    def plotObs(self):
        plt.plot(self.t[self.outlier==True], self.m[self.outlier==True],
                 'o',color='0.75',mec='none')
        plt.plot(self.t[self.outlier==False],self.m[self.outlier==False],
                 'ko')
        plt.errorbar(self.t, self.m, self.merr,fmt='none',ecolor="0.8")
        plt.savefig("lightcurve.png")

    def plot_phase(self, period):
        phase = self.t/period % 1
        plt.errorbar(phase, self.m, self.merr, fmt='o')
        plt.show()



