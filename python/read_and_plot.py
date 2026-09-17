import matplotlib
matplotlib.use('agg')
from matplotlib import rc

import matplotlib.pyplot as plt


import numpy as np


def IWC_extinction(P,extinction):
   return 10**(P[1])*extinction**P[0]


fp = open('../data/AustralianCirrusClouds.csv','r')


data=fp.readlines()


fp.close()


# convert text string to arrays of numbers
length_array=len(data)-1

times=np.zeros((length_array))
extinction=np.zeros((length_array))
ice_water=np.zeros((length_array))

for i in range(length_array):
	string1 = data[i+1]

	columns = string1.split(', ')

	times[i] = float(columns[0])
	extinction[i]=float(columns[1])
	ice_water[i]=float(columns[2].replace('\n',''))




plt.plot(np.log10(extinction), np.log10(ice_water), '.' ) 

plt.savefig('../output/myplot.png')


# filter the data for bad datapoints
ind,=np.where((ice_water>0) & (extinction>0))  

logIWC=np.log10(ice_water)    
logExt=np.log10(extinction) 

cc=np.corrcoef(logIWC[ind], logExt[ind])   
print (cc[0][1])

# do linear fit
P=np.polyfit(logExt[ind], logIWC[ind],1)  

# equation of extinction vs IWC
print(IWC_extinction(P, 1.e-5))

fig=plt.figure()
plt.plot(extinction, ice_water,'.')


ext1=np.linspace(0,1e-3, 100)
plt.plot(ext1, IWC_extinction(P, ext1))

plt.text(0.0002,0.010,'$IWC = ' + str(round(10**P[1],2)) + '\\times Ext^{' + str(round(P[0],3)) + '}$' )

plt.title('A plot of data and regression curve fit')
plt.xlabel('Extinction')
plt.ylabel('Ice water content (g m$^{-3}$)')
plt.savefig('../output/myplot2.png')

# pearsons correlation coefficient
r=cc[0][1]
# test for significance
n=len(logExt[ind])
t1=r/np.sqrt((1.-r**2)/(n-2))

from scipy.stats import t   
tcrit = np.abs(t.ppf(0.05/2., n-2))
if (tcrit < t1):
	print('The correlation is significant')
else:
	print('The correlation is not significant')




# read in the martian clouds dataset
fp = open('../data/martian_clouds.csv','r')


data=fp.readlines()


fp.close()


# convert text string to arrays of numbers
length_array=len(data)-2

height=np.zeros((length_array))
dust=np.zeros((length_array))
cloud=np.zeros((length_array))

for i in range(length_array):
        string1 = data[i+2]

        columns = string1.split(', ')

        height[i] = float(columns[0])
        dust[i]=float(columns[1])
        cloud[i]=float(columns[2].replace('\n',''))

# calculate the ice water content on Mars - our prediction
IWC = IWC_extinction(P,  cloud)
# plot to a figure
plt.figure()
plt.plot(IWC, height)


plt.xlabel('Ice Water Content (g m$^{-3}$)')
plt.ylabel('Height (m)')
plt.savefig('../output/myplot3.png')

# calculate statistics of IWC
ind,=np.where((height>=3000 ) & (height<=4000))
print(np.mean(IWC[ind]))
print(np.std(IWC[ind] ))

print(np.std(IWC[ind] )/np.sqrt(len(ind)))

plt.errorbar(np.mean(IWC[ind]), np.mean(height[ind]), \
  np.std(height[ind])/np.sqrt(len(ind)), np.std(IWC[ind])/np.sqrt(len(ind)))


plt.savefig('../output/myplot4.png')


# Ice water path (g m-2)
print(np.sum(IWC[1:]*np.diff(height)) )





# zoomed plot / inset 

plt.figure()
plt.plot(IWC, height, linewidth=4)
plt.errorbar(np.mean(IWC[ind]), np.mean(height[ind]), \
  np.std(height[ind])/np.sqrt(len(ind)), np.std(IWC[ind])/np.sqrt(len(ind)))


p1=np.percentile(IWC[ind],50)
p2=np.percentile(IWC[ind],25)
p3=np.percentile(IWC[ind],75)

print(p1)
print(p2)
print(p3)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)


plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False)
plt.ylim((2000,4000))
plt.savefig('../output/myplot5.png')


