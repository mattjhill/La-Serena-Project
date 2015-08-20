from lightcurve import *

testCurve = LightCurve("2110217026031.dat")
testCurve.lowessClean(K=2)
print testCurve.t
testCurve.plotObs()



