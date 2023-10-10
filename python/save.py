res_lsq, err, infodict, errmsg, success = leastsq(fun, [A, b], args=((extinction[ind]),(ice_water[ind])),full_output=1)
from scipy.optimize import leastsq
res_lsq[1]+pcov[1][1]**0.5
https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
