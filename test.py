from lightcurve import *

testCurve = LightCurve("2110217026031.dat")
testCurve.lowessClean(0.2)
testCurve.plotObs()



